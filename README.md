# LLM Finetuning: DDP vs FSDP with SFTTrainer + Accelerate

Finetune **Qwen2.5-7B** on the **Alpaca** dataset using HuggingFace SFTTrainer.
Same training logic, two distributed strategies — just swap the accelerate config file.

This project demonstrates a key advantage of FSDP over DDP: **FSDP can train with much larger batch sizes** because it shards model state across GPUs, freeing per-GPU memory for activations.

## Prerequisites

- 4× NVIDIA GPUs (tested on A100 80GB)
- Python 3.10+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager

## Setup

```bash
# Install dependencies
uv sync
```

## Understanding DDP vs FSDP

### The Core Difference

Both DDP and FSDP are strategies for training across multiple GPUs, but they manage GPU memory very differently.

**DDP (Distributed Data Parallel)** — each GPU holds a **full copy** of everything:

```
GPU 0: [Full Model 14GB] + [Full Optimizer 28GB] + [Activations ???]
GPU 1: [Full Model 14GB] + [Full Optimizer 28GB] + [Activations ???]
GPU 2: [Full Model 14GB] + [Full Optimizer 28GB] + [Activations ???]
GPU 3: [Full Model 14GB] + [Full Optimizer 28GB] + [Activations ???]
```

**FSDP (Fully Sharded Data Parallel)** — model and optimizer are **sharded** (split) across GPUs:

```
GPU 0: [1/4 Model 3.5GB] + [1/4 Optimizer 7GB] + [Activations !!!!!!]
GPU 1: [1/4 Model 3.5GB] + [1/4 Optimizer 7GB] + [Activations !!!!!!]
GPU 2: [1/4 Model 3.5GB] + [1/4 Optimizer 7GB] + [Activations !!!!!!]
GPU 3: [1/4 Model 3.5GB] + [1/4 Optimizer 7GB] + [Activations !!!!!!]
```

### Memory Math with Qwen2.5-7B on 80GB GPUs

| Component | DDP (per GPU) | FSDP (per GPU) |
|---|---|---|
| Model parameters (bf16) | ~14 GB | ~3.5 GB (÷4 GPUs) |
| Optimizer states (Adam) | ~28 GB | ~7 GB (÷4 GPUs) |
| Gradients (bf16) | ~14 GB | ~3.5 GB (÷4 GPUs) |
| **Total fixed cost** | **~56 GB** | **~14 GB** |
| **Free for activations** | **~24 GB** | **~66 GB** |

> **Why this matters:** Activations scale with batch size. More free memory = larger batches = faster training. With DDP you're limited to batch size 2 per GPU. With FSDP you can go to 8 or higher — a 4× throughput improvement.

### How DDP Works

1. Each GPU loads a **full copy** of the model
2. Each GPU processes a different mini-batch (data parallelism)
3. After the backward pass, gradients are **all-reduced** (averaged) across GPUs
4. Each GPU updates its own copy of the model with the same averaged gradients

**Pros:** Simple, low communication overhead
**Cons:** Model must fit entirely on one GPU (with optimizer + gradients + activations)

### How FSDP Works

1. Model parameters are **sharded** — each GPU holds only 1/N of the parameters
2. Before each forward/backward pass, parameters are **all-gathered** (temporarily reassembled)
3. After computing gradients, they are **reduce-scattered** (averaged and re-sharded)
4. Each GPU only updates its own shard of the optimizer states

**Pros:** Dramatically lower per-GPU memory usage
**Cons:** Higher communication overhead (all-gather before each layer)

### When to Use What

| | DDP | FSDP |
|---|---|---|
| **Memory per GPU** | Full model + optimizer + gradients | 1/N of model + optimizer + gradients |
| **Communication** | All-reduce gradients (once per step) | All-gather params + reduce-scatter gradients (per layer) |
| **Best for** | Models that comfortably fit on a single GPU | Large models that need memory optimization |
| **Batch size** | Limited by leftover GPU memory | Much larger — sharding frees memory |
| **Overhead** | Lower | Higher (more communication) |
| **Complexity** | Simpler setup | More config options |

**Rule of thumb:** If your model + optimizer + gradients fit on one GPU with plenty of room for activations, DDP is simpler and faster. If you're memory-constrained, FSDP lets you trade communication overhead for larger batches (or train models that wouldn't fit at all with DDP).

## Accelerate Configuration

Accelerate configs live in `configs/` and control the distributed strategy.
You can either use the provided YAML files directly or generate your own via `accelerate config`.

### DDP Config

`configs/ddp_config.yaml`:
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_machines: 1
num_processes: 4
mixed_precision: bf16
main_training_function: main
```

Key fields:
- `distributed_type: MULTI_GPU` — tells accelerate to use DDP
- `num_processes: 4` — number of GPUs (one process per GPU)
- `mixed_precision: bf16` — uses bfloat16 for faster training and lower memory

To generate interactively:
```bash
accelerate config --config_file configs/ddp_config.yaml
# Compute environment → This machine
# Distributed training → multi-GPU
# How many machines → 1
# How many GPUs → 4
# Mixed precision → bf16
```

### FSDP Config

`configs/fsdp_config.yaml`:
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
num_machines: 1
num_processes: 4
mixed_precision: bf16
main_training_function: main
fsdp_config:
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_transformer_layer_cls_to_wrap: Qwen2DecoderLayer
  fsdp_backward_prefetch_policy: BACKWARD_PRE
  fsdp_forward_prefetch: false
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_use_orig_params: true
  fsdp_cpu_ram_efficient_loading: true
  fsdp_offload_params: false
```

Key FSDP fields explained:

| Field | Value | What it does |
|---|---|---|
| `fsdp_sharding_strategy` | `FULL_SHARD` | Shards parameters, gradients, AND optimizer states (maximum memory savings) |
| `fsdp_auto_wrap_policy` | `TRANSFORMER_BASED_WRAP` | Wraps each transformer layer as a separate FSDP unit |
| `fsdp_transformer_layer_cls_to_wrap` | `Qwen2DecoderLayer` | The layer class to wrap — **must match your model architecture** |
| `fsdp_backward_prefetch_policy` | `BACKWARD_PRE` | Prefetches next layer's params during backward pass (faster) |
| `fsdp_state_dict_type` | `SHARDED_STATE_DICT` | Saves checkpoints in sharded format (avoids gathering full model to one GPU) |
| `fsdp_sync_module_states` | `true` | Ensures all GPUs start with identical weights |
| `fsdp_use_orig_params` | `true` | Required for compatibility with HuggingFace Trainer |
| `fsdp_cpu_ram_efficient_loading` | `true` | Loads model on CPU first, then distributes (avoids GPU OOM during init) |
| `fsdp_offload_params` | `false` | Set to `true` to offload params to CPU (slower but saves even more GPU memory) |

> **Important:** When switching models, you must update `fsdp_transformer_layer_cls_to_wrap` to match the new model's decoder layer class. For example: `LlamaDecoderLayer` for Llama models, `Qwen2DecoderLayer` for Qwen2.5 models, `MistralDecoderLayer` for Mistral, etc.

To generate interactively:
```bash
accelerate config --config_file configs/fsdp_config.yaml
# Compute environment → This machine
# Distributed training → FSDP
# How many machines → 1
# How many GPUs → 4
# Sharding strategy → FULL_SHARD
# Auto wrap policy → TRANSFORMER_BASED_WRAP
# Transformer layer class → Qwen2DecoderLayer
# Backward prefetch → BACKWARD_PRE
# State dict type → SHARDED_STATE_DICT
# Mixed precision → bf16
```

## Run Training

### DDP (batch size = 2 per GPU)

```bash
accelerate launch --config_file configs/ddp_config.yaml train_ddp.py
```

Effective batch size: 2 × 4 (grad accum) × 4 (GPUs) = **32**

### FSDP (batch size = 8 per GPU)

```bash
accelerate launch --config_file configs/fsdp_config.yaml train_fsdp.py
```

Effective batch size: 8 × 4 (grad accum) × 4 (GPUs) = **128**

> **The key takeaway:** FSDP handles a 4× larger per-device batch size than DDP on the same hardware. DDP would run out of memory (OOM) at batch size 8 with Qwen2.5-7B on 80GB GPUs, while FSDP handles it comfortably.

### Experimenting with Batch Sizes

To see the OOM difference yourself, try editing the batch size in `train_ddp.py`:

```python
# In train_ddp.py — change this:
BATCH_SIZE = 2
# To this (will likely OOM):
BATCH_SIZE = 8
```

You should see a CUDA out-of-memory error with DDP, proving that FSDP's memory savings are real and meaningful.

## Project Structure

```
├── configs/
│   ├── ddp_config.yaml       # Accelerate config for DDP
│   └── fsdp_config.yaml      # Accelerate config for FSDP
├── src/
│   ├── __init__.py
│   ├── data.py               # Dataset loading & Alpaca formatting
│   ├── train.py              # SFTTrainer training logic + hyperparameters
│   └── utils.py              # Utilities
├── train_ddp.py              # DDP entry point (batch_size=2)
├── train_fsdp.py             # FSDP entry point (batch_size=8)
├── pyproject.toml            # Dependencies (managed by uv)
└── README.md
```

## Configuration

Hyperparameters are defined in `src/train.py`. Batch size is set per-strategy in the entry point scripts.

| Parameter | Default | Description |
|---|---|---|
| `MODEL_NAME` | `Qwen/Qwen2.5-7B` | Model to finetune |
| `DEFAULT_BATCH_SIZE` | 2 | Per-device batch size (overridden by entry scripts) |
| `GRAD_ACCUM_STEPS` | 4 | Gradient accumulation steps |
| `MAX_SEQ_LENGTH` | 512 | Max token length |
| `NUM_SAMPLES` | 10,000 | Subset size from Alpaca dataset |
| `NUM_EPOCHS` | 1 | Number of training epochs |
| `LR` | 2e-5 | Learning rate |
| `save_model` | `False` | Whether to save the trained model and tokenizer to disk |

> **Disk space:** By default, model saving is disabled (`save_model=False`) and no checkpoints are written during training. To save the final model, pass `save_model=True` to the `train()` function in your entry script:
> ```python
> train(batch_size=BATCH_SIZE, save_model=True)
> ```

### Batch Sizes Per Strategy

| | Per-device | × Grad Accum | × GPUs | Effective |
|---|---|---|---|---|
| **DDP** | 2 | 4 | 4 | **32** |
| **FSDP** | 8 | 4 | 4 | **128** |
