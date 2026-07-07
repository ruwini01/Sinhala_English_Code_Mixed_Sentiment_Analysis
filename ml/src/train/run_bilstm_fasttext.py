"""B2 / bilstm_fasttext: BiLSTM over fastText Sinhala embeddings (cc.si.300).

Pre-transformer neural floor. Embeddings initialized from fastText Sinhala
.bin (subword model, so romanized/OOV tokens still get vectors), fine-tuned
during training. Class-weighted CE, early stopping on val macro-F1, seed=42.

Run from ml/ (Colab recommended — loading the .bin needs ~8 GB RAM):
    python -m src.train.run_bilstm_fasttext --fasttext data/embeddings/cc.si.300.bin
"""

import argparse
import time

import fasttext
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.common.schema import NUM_CLASSES, SEED, TEXT_COL
from src.train.data_io import load_split_frames
from src.train.results_io import evaluate, save_results

HP = {
    "max_len": 64,
    "min_freq": 2,
    "emb_dim": 300,
    "hidden": 256,
    "layers": 1,
    "dropout": 0.3,
    "batch_size": 64,
    "lr": 1e-3,
    "epochs": 15,
    "patience": 3,
}
PAD, UNK = 0, 1


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class BiLSTM(nn.Module):
    def __init__(self, emb_matrix):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(
            torch.tensor(emb_matrix, dtype=torch.float), freeze=False, padding_idx=PAD
        )
        self.lstm = nn.LSTM(HP["emb_dim"], HP["hidden"], num_layers=HP["layers"],
                            batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(HP["dropout"])
        self.fc = nn.Linear(HP["hidden"] * 4, NUM_CLASSES)  # mean-pool + max-pool

    def forward(self, x):
        mask = (x != PAD).unsqueeze(-1).float()
        out, _ = self.lstm(self.emb(x))
        mean = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        maxp = out.masked_fill(mask == 0, -1e9).max(1).values
        return self.fc(self.drop(torch.cat([mean, maxp], dim=1)))


def encode(texts, vocab):
    rows = []
    for t in texts:
        ids = [vocab.get(w, UNK) for w in str(t).split()][: HP["max_len"]]
        rows.append(ids + [PAD] * (HP["max_len"] - len(ids)))
    return torch.tensor(rows, dtype=torch.long)


def predict(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for (xb,) in loader:
            preds.extend(model(xb.to(device)).argmax(1).cpu().tolist())
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasttext", default="data/embeddings/cc.si.300.bin")
    args = ap.parse_args()

    set_seed(SEED)
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits = load_split_frames()

    # vocab from train split only
    freq = {}
    for t in splits["train"][TEXT_COL].fillna(""):
        for w in str(t).split():
            freq[w] = freq.get(w, 0) + 1
    words = [w for w, c in sorted(freq.items()) if c >= HP["min_freq"]]
    vocab = {w: i + 2 for i, w in enumerate(words)}  # 0=PAD, 1=UNK

    ft = fasttext.load_model(args.fasttext)
    rng = np.random.default_rng(SEED)
    emb = rng.normal(0, 0.1, (len(vocab) + 2, HP["emb_dim"])).astype("float32")
    emb[PAD] = 0.0
    for w, i in vocab.items():
        emb[i] = ft.get_word_vector(w)
    del ft  # free ~7 GB

    X = {k: encode(v[TEXT_COL].fillna(""), vocab) for k, v in splits.items()}
    y = {k: torch.tensor(v["label_id"].tolist()) for k, v in splits.items()}
    train_dl = DataLoader(TensorDataset(X["train"], y["train"]),
                          batch_size=HP["batch_size"], shuffle=True,
                          generator=torch.Generator().manual_seed(SEED))
    val_dl = DataLoader(TensorDataset(X["val"]), batch_size=256)
    test_dl = DataLoader(TensorDataset(X["test"]), batch_size=256)

    model = BiLSTM(emb).to(device)
    counts = np.bincount(y["train"], minlength=NUM_CLASSES)
    weights = torch.tensor(len(y["train"]) / (NUM_CLASSES * counts),
                           dtype=torch.float).to(device)
    crit = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters(), lr=HP["lr"])

    best_f1, best_state, patience = -1.0, None, 0
    for epoch in range(HP["epochs"]):
        model.train()
        for xb, yb in train_dl:
            opt.zero_grad()
            loss = crit(model(xb.to(device)), yb.to(device))
            loss.backward()
            opt.step()
        val_f1 = evaluate(y["val"].tolist(), predict(model, val_dl, device))["macro_f1"]
        print(f"epoch {epoch + 1:2d}  val macro-F1 {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1, patience = val_f1, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= HP["patience"]:
                print("early stop")
                break

    model.load_state_dict(best_state)
    model.to(device)
    val_metrics = evaluate(y["val"].tolist(), predict(model, val_dl, device))
    test_metrics = evaluate(y["test"].tolist(), predict(model, test_dl, device))

    save_results(
        experiment_id="bilstm_fasttext",
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
        wall_clock_sec=time.time() - t0,
        hyperparams={**HP, "vocab_size": len(vocab) + 2, "device": device},
        notes="B2 pre-transformer floor; fastText cc.si.300 init, trainable embeddings",
    )


if __name__ == "__main__":
    main()