"""P1 / lora_scl_lid: the proposed model (claims 1 + 3).

loss = CrossEntropy(weighted) + lambda * SupConLoss   (lambda=0.1 default)

Loads the one-time tokenized cache (data/processed/tokenized/*.pt) which
carries per-token LID tag ids. Saves the best checkpoint for the
explainability stage. Records per-epoch history (train_loss, val_loss,
val_macro_f1) into the results JSON for the curve figures, plus a binary
positive-vs-negative evaluation of the same trained model.

Training stability: linear warmup (10%) + decay schedule, gradient clipping
(max-norm 1.0), mixed precision on GPU.

Run from ml/ (Colab T4):
    python -m src.train.run_lora_scl_lid --epochs 18 --patience 4
Ablations reuse this entrypoint:
    python -m src.train.run_lora_scl_lid --id abl_no_scl --lam 0
    python -m src.train.run_lora_scl_lid --id abl_no_lid --no-lid
    python -m src.train.run_lora_scl_lid --id abl_lora_only --lam 0 --no-lid
    python -m src.train.run_lora_scl_lid --id abl_no_lora --no-lora --lr 2e-5
    python -m src.train.run_lora_scl_lid --id abl_rank_4 --rank 4
    python -m src.train.run_lora_scl_lid --id abl_rank_16 --rank 16
    python -m src.train.run_lora_scl_lid --id abl_q_only --targets query
Larger backbone (fits T4 only because of LoRA):
    python -m src.train.run_lora_scl_lid --id lora_scl_lid_large \
        --model xlm-roberta-large --batch-size 16 --lr 1e-4
Different seed (variance reporting):
    python -m src.train.run_lora_scl_lid --seed 43
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_linear_schedule_with_warmup

from src.common.schema import LABEL2ID, NUM_CLASSES, SEED, TOKENIZED_DIR
from src.models.proposed_xlmr_lora import ProposedModel
from src.models.scl_loss import SupConLoss
from src.train.results_io import evaluate, save_results

CKPT_DIR = Path("checkpoints")
NEUTRAL = LABEL2ID["neutral"]


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cache(name):
    d = torch.load(TOKENIZED_DIR / f"{name}.pt", weights_only=False)
    return d["input_ids"], d["attention_mask"], d["lid_ids"], d["labels"]


def collect_logits(model, loader, device):
    model.eval()
    chunks = []
    with torch.no_grad():
        for ids, attn, lid in loader:
            logits, _ = model(ids.to(device), attn.to(device), lid.to(device))
            chunks.append(logits.cpu())
    return torch.cat(chunks)


def binary_pos_neg(logits, y_true):
    """Positive-vs-negative subtask on the SAME trained model: neutral rows
    are FILTERED (never relabeled); prediction = argmax over the two
    polarity logits only."""
    mask = y_true != NEUTRAL
    polar = [LABEL2ID["negative"], LABEL2ID["positive"]]
    preds = logits[mask][:, polar].argmax(1).tolist()          # 0=neg, 1=pos
    truth = (y_true[mask] == LABEL2ID["positive"]).long().tolist()
    return {
        "accuracy": float(accuracy_score(truth, preds)),
        "macro_f1": float(f1_score(truth, preds, average="macro")),
        "n_rows": int(mask.sum()),
        "note": "neutral rows filtered at evaluation; labels never remapped",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="lora_scl_lid")
    ap.add_argument("--model", default="xlm-roberta-base")
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--no-lid", action="store_true")
    ap.add_argument("--no-lora", action="store_true")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--targets", nargs="+", default=["query", "value"])
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--warmup-frac", type=float, default=0.1)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    set_seed(args.seed)
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    print(f"device: {device}")

    data = {name: load_cache(name) for name in ("train", "val", "test")}
    y = {k: v[3] for k, v in data.items()}
    train_dl = DataLoader(TensorDataset(*data["train"]), batch_size=args.batch_size,
                          shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    val_dl = DataLoader(TensorDataset(*data["val"][:3]), batch_size=128)
    test_dl = DataLoader(TensorDataset(*data["test"][:3]), batch_size=128)

    model = ProposedModel(use_lora=not args.no_lora, use_lid=not args.no_lid,
                          lora_r=args.rank, lora_alpha=args.alpha,
                          target_modules=args.targets,
                          model_name=args.model).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    counts = np.bincount(y["train"].numpy(), minlength=NUM_CLASSES)
    class_w = torch.tensor(len(y["train"]) / (NUM_CLASSES * counts),
                           dtype=torch.float).to(device)
    ce = nn.CrossEntropyLoss(weight=class_w)
    scl = SupConLoss(temperature=args.temperature)
    trainable_ps = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_ps, lr=args.lr, weight_decay=0.01)
    total_steps = len(train_dl) * args.epochs
    sched = get_linear_schedule_with_warmup(
        opt, int(total_steps * args.warmup_frac), total_steps)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    def val_ce_loss(logits_val):
        return float(ce(logits_val.to(device), y["val"].to(device)).cpu())

    history = {"train_loss": [], "val_loss": [], "val_macro_f1": []}
    best_f1, best_state, patience = -1.0, None, 0
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        for ids, attn, lid, yb in train_dl:
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, feats = model(ids.to(device), attn.to(device), lid.to(device))
                yb = yb.to(device)
                loss = ce(logits, yb) + args.lam * scl(feats.float(), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable_ps, max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            epoch_losses.append(float(loss.detach().cpu()))

        val_logits = collect_logits(model, val_dl, device)
        val_f1 = evaluate(y["val"].tolist(), val_logits.argmax(1).tolist())["macro_f1"]
        train_loss = float(np.mean(epoch_losses))
        vloss = val_ce_loss(val_logits)
        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(vloss, 4))
        history["val_macro_f1"].append(round(val_f1, 4))
        print(f"epoch {epoch + 1:2d}  train loss {train_loss:.4f}  "
              f"val loss {vloss:.4f}  val macro-F1 {val_f1:.4f}")

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

    val_logits = collect_logits(model, val_dl, device)
    test_logits = collect_logits(model, test_dl, device)
    val_metrics = evaluate(y["val"].tolist(), val_logits.argmax(1).tolist())
    test_metrics = evaluate(y["test"].tolist(), test_logits.argmax(1).tolist())
    test_loss = float(ce(test_logits.to(device), y["test"].to(device)).cpu())
    binary = binary_pos_neg(test_logits, y["test"])
    print(f"binary pos/neg: accuracy {binary['accuracy']:.4f}  "
          f"macro-F1 {binary['macro_f1']:.4f}  ({binary['n_rows']} rows)")

    save_results(
        experiment_id=args.id,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        trainable_params=trainable,
        wall_clock_sec=time.time() - t0,
        hyperparams={**vars(args), "device": device, "max_len": 128},
        notes="P1 proposed: XLM-R frozen + LoRA + SCL + LID (claims 1+3)",
        history=history,
        extra={"test_loss": round(test_loss, 4), "binary_pos_neg": binary},
    )


if __name__ == "__main__":
    main()
