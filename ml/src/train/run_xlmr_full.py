"""B4 / xlmr_full: XLM-R full fine-tune baseline.

This is the comparison target for the proposed LoRA model (claim 1):
trainable_params and wall_clock_sec recorded here ARE the efficiency evidence.

Run from ml/ (Colab T4):
    python -m src.train.run_xlmr_full
"""

from src.train.hf_trainer import run_full_finetune

if __name__ == "__main__":
    run_full_finetune(
        experiment_id="xlmr_full",
        model_name="xlm-roberta-base",
        notes="B4 full fine-tune; LoRA efficiency comparison target (claim 1)",
        hp_overrides={"epochs": 8, "patience": 3, "lr": 3e-5},
    )