"""B3 / mbert_full: mBERT full fine-tune baseline.

Run from ml/ (Colab T4):
    python -m src.train.run_mbert_full
"""

from src.train.hf_trainer import run_full_finetune

if __name__ == "__main__":
    run_full_finetune(
        experiment_id="mbert_full",
        model_name="bert-base-multilingual-cased",
        notes="B3 backbone comparison; full fine-tune, weighted CE",
        hp_overrides={"epochs": 8, "patience": 3, "lr": 3e-5},
    )