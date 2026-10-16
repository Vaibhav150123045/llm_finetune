"""
FSDP (Fully Sharded Data Parallel) training entry point.

Model parameters, gradients, and optimizer states are sharded across GPUs.
With Qwen2.5-7B, each GPU only holds ~10.5GB (1/4 of model + optimizer),
leaving ~70GB for activations — enabling much larger batch sizes.

Usage:
    accelerate launch --config_file configs/fsdp_config.yaml train_fsdp.py
"""

from src.train import train

# FSDP: 4x larger batch size — sharding frees up ~30GB per GPU
BATCH_SIZE = 8

if __name__ == "__main__":
    train(batch_size=BATCH_SIZE)
