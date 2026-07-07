"""P1 / lora_scl_lid: the proposed model (claims 1 + 3).

loss = CrossEntropy(weighted) + lambda * SupConLoss   (lambda=0.1 default)

Loads the one-time tokenized cache (data/processed/tokenized/*.pt) which
carries per-token LID tag ids. Saves the best checkpoint for the
explainability stage.

Run from ml/ (Colab T4):
    python -m src.train.run_lora_scl_lid
Ablations reuse this entrypoint:
    python -m src.train.run_lora_scl_lid --id abl_no_scl --lam 0
    python -m src.train.run_lora_scl_lid --id abl_no_lid --no-lid
    python -m src.train.run_lora_scl_lid --id abl_lora_only --lam 0 --no-lid
    python -m src.train.run_lora_scl_lid --id abl_no_lora --no-lora --lr 2e-5
    python -m src.train.run_lora_scl_lid --id abl_rank_4 --rank 4
    python -m src.train.run_lora_scl_lid --id abl_q_only --targets query
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.common.schema import NUM_CLASSES, SEED, TOKENIZED_DIR
from src.models.proposed_xlmr_lora import ProposedModel
from src.models.scl_loss import SupConLoss
from src.train.results_io import evaluate, save_results

CKPT_DIR = Path("checkpoints")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cache(name):
    d = torch.load(TOKENIZED_DIR / f"{name}.pt", weights_only=False)
    return d["input_ids"], d["attention_mask"], d["lid_ids"], d["labels"]


def predict(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for ids, attn, lid in loader:
            logits, _ = model(ids.to(device), attn.to(device), lid.to(device))
            preds.extend(logits.argmax(1).cpu().tolist())
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="lora_scl_lid")
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--no-lid", action="store_true")
    ap.add_argument("--no-lora", action="store_true")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--targets", nargs="+", default=["query", "value"])
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.07)
    args = ap.parse_args()

    set_seed(SEED)
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    data = {name: load_cache(name) for name in ("train", "val", "test")}
    y = {k: v[3] for k, v in data.items()}
    train_dl = DataLoader(TensorDataset(*data["train"]), batch_size=args.batch_size,
                          shuffle=True, generator=torch.Generator().manual_seed(SEED))
    val_dl = DataLoader(TensorDataset(*data["val"][:3]), batch_size=128)
    test_dl = DataLoader(TensorDataset(*data["test"][:3]), batch_size=128)

    model = ProposedModel(use_lora=not args.no_lora, use_lid=not args.no_lid,
                          lora_r=args.rank, lora_alpha=args.alpha,
                          target_modules=args.targets).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    counts = np.bincount(y["train"].numpy(), minlength=NUM_CLASSES)
    class_w = torch.tensor(len(y["train"]) / (NUM_CLASSES * counts),
                           dtype=torch.float).to(device)
    ce = nn.CrossEntropyLoss(weight=class_w)
    scl = SupConLoss(temperature=args.temperature)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.01)

    best_f1, best_state, patience = -1.0, None, 0
    for epoch in range(args.epochs):
        model.train()
        for ids, attn, lid, yb in train_dl:
            opt.zero_grad()
            logits, feats = model(ids.to(device), attn.to(device), lid.to(device))
            yb = yb.to(device)
            loss = ce(logits, yb) + args.lam * scl(feats, yb)
            loss.backward()
            opt.step()
        val_f1 = evaluate(y["val"].tolist(), predict(model, val_dl, device))["macro_f1"]
        print(f"epoch {epoch + 1:2d}  val macro-F1 {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1, patience = val_f1, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= args.patience:
                print("early stop")
                break

    model.load_state_dict(best_state)
    model.to(device)
    CKPT_DIR.mkdir(exist_ok=True)
    torch.save(best_state, CKPT_DIR / f"{args.id}.pt")
    print(f"checkpoint -> {CKPT_DIR / (args.id + '.pt')}")

    val_metrics = evaluate(y["val"].tolist(), predict(model, val_dl, device))
    test_metrics = evaluate(y["test"].tolist(), predict(model, test_dl, device))
    save_results(
        experiment_id=args.id,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        trainable_params=trainable,
        wall_clock_sec=time.time() - t0,
        hyperparams={**vars(args), "seed": SEED, "device": device,
                     "base_model": "xlm-roberta-base", "max_len": 128},
        notes="P1 proposed: XLM-R frozen + LoRA + SCL + LID (claims 1+3)",
    )


if __name__ == "__main__":
    main()