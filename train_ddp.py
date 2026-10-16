"""
DDP (Distributed Data Parallel) training entry point.

Each GPU holds a full copy of the model and optimizer states.
With Qwen2.5-7B, each GPU must hold ~42GB (model + optimizer),
leaving ~38GB for activations — limiting per-device batch size.

Usage:
    accelerate launch --config_file configs/ddp_config.yaml train_ddp.py
"""

from src.train import train

# DDP: conservative batch size — full model replica on every GPU
BATCH_SIZE = 2

if __name__ == "__main__":
    train(batch_size=BATCH_SIZE)
