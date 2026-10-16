"""Core training logic using HuggingFace SFTTrainer."""

import torch
from transformers import AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

from src.data import load_tokenizer, get_dataset

# ── Hyperparameters ──────────────────────────────────────────────────────────
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET_NAME = "yahma/alpaca-cleaned"
MAX_SEQ_LENGTH = 512
DEFAULT_BATCH_SIZE = 2   # per device (conservative default for DDP)
GRAD_ACCUM_STEPS = 4
NUM_EPOCHS = 1
LR = 2e-5
SEED = 42
LOG_EVERY = 50
NUM_SAMPLES = 10_000
OUTPUT_DIR = "output"


def train(batch_size=DEFAULT_BATCH_SIZE, save_model=False):
    tokenizer = load_tokenizer(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)

    dataset = get_dataset(DATASET_NAME, tokenizer, MAX_SEQ_LENGTH, NUM_SAMPLES)

    effective_batch = batch_size * GRAD_ACCUM_STEPS * 4  # 4 GPUs
    print(f"Per-device batch size: {batch_size}")
    print(f"Effective batch size:  {effective_batch}")

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LR,
        bf16=False,
        logging_steps=LOG_EVERY,
        save_strategy="no",
        seed=SEED,
        max_length=MAX_SEQ_LENGTH,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    if save_model:
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"Model saved to {OUTPUT_DIR}/")
    else:
        print("Skipping model save (save_model=False)")


if __name__ == "__main__":
    train(save_model=True)
