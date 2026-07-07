"""Shared full fine-tuning trainer for transformer baselines (B3 mBERT, B4 XLM-R).

Tokenizes clean_text on the fly with the model's own tokenizer (mBERT cannot
consume the XLM-R cache; both baselines use identical settings — max_len=128,
locked splits, weighted CE, early stopping on val macro-F1, seed=42 — so the
backbone comparison is fair). fp16 on GPU.
"""

import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src.common.schema import NUM_CLASSES, SEED, TEXT_COL
from src.train.data_io import load_split_frames
from src.train.results_io import evaluate, save_results

DEFAULT_HP = {
    "max_len": 128,
    "batch_size": 16,
    "lr": 2e-5,
    "epochs": 4,
    "patience": 2,
    "warmup_frac": 0.1,
    "weight_decay": 0.01,
}


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _predict(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for input_ids, attn in loader:
            logits = model(input_ids=input_ids.to(device),
                           attention_mask=attn.to(device)).logits
            preds.extend(logits.argmax(1).cpu().tolist())
    return preds


def run_full_finetune(experiment_id, model_name, notes, hp_overrides=None):
    hp = {**DEFAULT_HP, **(hp_overrides or {}), "model_name": model_name}
    set_seed(SEED)
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"

    splits = load_split_frames()
    tok = AutoTokenizer.from_pretrained(model_name)
    enc = {
        k: tok(list(v[TEXT_COL].fillna("")), truncation=True,
               max_length=hp["max_len"], padding="max_length", return_tensors="pt")
        for k, v in splits.items()
    }
    y = {k: torch.tensor(v["label_id"].tolist()) for k, v in splits.items()}

    train_dl = DataLoader(
        TensorDataset(enc["train"]["input_ids"], enc["train"]["attention_mask"], y["train"]),
        batch_size=hp["batch_size"], shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    val_dl = DataLoader(TensorDataset(enc["val"]["input_ids"], enc["val"]["attention_mask"]),
                        batch_size=128)
    test_dl = DataLoader(TensorDataset(enc["test"]["input_ids"], enc["test"]["attention_mask"]),
                         batch_size=128)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=NUM_CLASSES).to(device)

    counts = np.bincount(y["train"], minlength=NUM_CLASSES)
    class_w = torch.tensor(len(y["train"]) / (NUM_CLASSES * counts),
                           dtype=torch.float).to(device)
    crit = nn.CrossEntropyLoss(weight=class_w)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"],
                            weight_decay=hp["weight_decay"])
    total_steps = len(train_dl) * hp["epochs"]
    sched = get_linear_schedule_with_warmup(
        opt, int(total_steps * hp["warmup_frac"]), total_steps)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_f1, best_state, patience = -1.0, None, 0
    for epoch in range(hp["epochs"]):
        model.train()
        for input_ids, attn, yb in train_dl:
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(input_ids=input_ids.to(device),
                               attention_mask=attn.to(device)).logits
                loss = crit(logits, yb.to(device))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        val_f1 = evaluate(y["val"].tolist(), _predict(model, val_dl, device))["macro_f1"]
        print(f"epoch {epoch + 1}  val macro-F1 {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1, patience = val_f1, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= hp["patience"]:
                print("early stop")
                break

    model.load_state_dict(best_state)
    model.to(device)
    val_metrics = evaluate(y["val"].tolist(), _predict(model, val_dl, device))
    test_metrics = evaluate(y["test"].tolist(), _predict(model, test_dl, device))

    save_results(
        experiment_id=experiment_id,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
        wall_clock_sec=time.time() - t0,
        hyperparams={**hp, "device": device},
        notes=notes,
    )