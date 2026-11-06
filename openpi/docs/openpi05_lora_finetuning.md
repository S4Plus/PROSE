# OpenPI05 LoRA Fine-tuning

This note summarizes how LoRA fine-tuning is implemented for OpenPI05 in this repo, which configs are used in practice, and what is actually trainable during "LoRA" training.

## Overview

OpenPI05 LoRA fine-tuning is built from three layers:

1. LoRA modules are defined in [`lora.py`](../src/openpi/models/lora.py).
2. Gemma model variants decide where LoRA is injected and with what rank.
3. Training configs decide which parameters are frozen through `freeze_filter`, which then determines `trainable_filter`.

For the common OpenPI05 LoRA configs in this repo, the model is:

- PaliGemma side: `gemma_2b_lora`
- Action expert side: `gemma_300m_lora`
- Policy type: `pi05=True`

Representative config:

- [`pi05_can_mh_lora`](../src/openpi/training/config.py)

## LoRA Implementation

LoRA is implemented in [`src/openpi/models/lora.py`](../src/openpi/models/lora.py).

### `LoRAConfig`

`LoRAConfig` defines:

- `rank`
- `alpha`
- `init_fn`
- `rslora`
- `axes`
- `label`

Scaling is:

- `alpha / rank` by default
- `alpha / sqrt(rank)` when `rslora=True`

### LoRA on attention weights

`lora.Einsum` wraps an einsum weight and optionally adds low-rank factors:

- base weight: `w`
- LoRA weights: `lora_a`, `lora_b`

Forward rule is:

`base(x, w) + lora(x, lora_a, lora_b) * scaling`

### LoRA on MLP weights

`lora.FeedForward` adds LoRA to:

- `gating_einsum`
- `linear`

with LoRA factors:

- `gating_einsum_lora_a`
- `gating_einsum_lora_b`
- `linear_lora_a`
- `linear_lora_b`

## Where LoRA Is Injected

Gemma LoRA variants are defined in [`src/openpi/models/gemma.py`](../src/openpi/models/gemma.py).

### Supported LoRA variants

- `gemma_2b_lora`
- `gemma_300m_lora`

### LoRA hyperparameters

- `gemma_2b_lora`
  - attention LoRA: `rank=16`, `alpha=16`
  - ffn LoRA: `rank=16`, `alpha=16`
- `gemma_300m_lora`
  - attention LoRA: `rank=32`, `alpha=32`
  - ffn LoRA: `rank=32`, `alpha=32`

### Attention injection points

LoRA is attached to the Gemma attention projections:

- `qkv_einsum`
- `q_einsum`
- `kv_einsum`
- `attn_vec_einsum`

### Feed-forward injection points

LoRA is attached to the Gemma MLP block:

- `mlp`

## OpenPI05 Model Structure

The OpenPI05 model is built in [`src/openpi/models/pi0.py`](../src/openpi/models/pi0.py).

At a high level it contains:

- `PaliGemma.llm`
- `PaliGemma.img`
- `action_in_proj`
- `action_out_proj`
- `time_mlp_in`
- `time_mlp_out`

For `pi05=True`, the action side uses `time_mlp_in/out` instead of the non-PI05 state/time projection path.

## Freeze Rules and What Is Actually Trainable

This is the most important practical detail.

Trainable parameters are defined by:

- `freeze_filter` in the train config
- `trainable_filter = nnx.All(nnx.Param, nnx.Not(freeze_filter))`

See [`src/openpi/training/config.py`](../src/openpi/training/config.py).

The default LoRA freeze logic lives in [`src/openpi/models/pi0_config.py`](../src/openpi/models/pi0_config.py).

### Current behavior

If `paligemma_variant` contains `"lora"`:

- freeze `.*llm.*`
- but exclude `.*lora.*` from freezing

If `action_expert_variant` contains `"lora"`:

- freeze the action-expert `llm` branch
- but exclude `.*lora.*` from freezing

### Important caveat

In this repo, "LoRA fine-tuning" is not strictly "LoRA-only training".

For configs such as `pi05_can_mh_lora`, the practical effect is:

- most `llm` base weights are frozen
- `llm` LoRA weights are trainable
- parameters outside `llm` are still trainable

That means modules such as:

- `PaliGemma.img.*`
- `action_in_proj.*`
- `action_out_proj.*`
- `time_mlp_in.*`
- `time_mlp_out.*`

can remain trainable in addition to LoRA parameters.

This matches what we observed in influence-analysis runs: trainable leaves were not only `*lora*`, but also image-side and action-head parameters.

If you want strict LoRA-only training, the current `freeze_filter` logic is not sufficient by itself and would need to be tightened.

## Common OpenPI05 LoRA Training Configs

The main OpenPI05 LoRA configs in [`src/openpi/training/config.py`](../src/openpi/training/config.py) are:

- `pi05_square_d1_lora`
- `pi05_can_mh_lora`
- `pi05_square_d1_core_lora_top80`
- `pi05_square_d1_core_lora_1000`

### `pi05_can_mh_lora`

This is the main config used in the recent experiments.

Key settings:

- model:
  - `pi05=True`
  - `paligemma_variant="gemma_2b_lora"`
  - `action_expert_variant="gemma_300m_lora"`
  - `action_horizon=20`
- dataset:
  - repo: `/workspace/liushuo/cupid/datasets/lerobot/can_mh`
  - delta joint actions enabled
- weights:
  - base checkpoint: `gs://openpi-assets/checkpoints/pi05_base/params`
- optimization:
  - `batch_size=32`
  - `num_train_steps=10000`
  - cosine schedule
  - warmup `1000`
  - peak lr `1e-4`
  - decay lr `1e-5`
- EMA:
  - `ema_decay=None`

## How Checkpoint Loading Works for LoRA

Weight loading is defined in [`src/openpi/training/weight_loaders.py`](../src/openpi/training/weight_loaders.py).

`CheckpointWeightLoader`:

1. loads the checkpoint params
2. merges them into the current model param tree
3. fills in missing keys that match `.*lora.*`

This is important because the base checkpoint does not contain newly added LoRA weights. Missing LoRA params are initialized from the current model definition and merged into the loaded state.

## How Training Uses the Trainable Filter

Training is implemented in [`scripts/train.py`](../scripts/train.py).

Key behavior:

- frozen params are converted to `bfloat16`
- optimizer state is created only for `params.filter(config.trainable_filter)`
- gradients are computed only for `config.trainable_filter`
- optimizer updates are applied only to the trainable subset

So the real meaning of "LoRA fine-tuning" in practice is whatever `trainable_filter` selects.

## What We Observed in Practice

For `pi05_can_mh_lora`, trainable leaves observed in analysis included:

- Gemma LoRA leaves under `PaliGemma/llm/.../lora_*`
- image-side parameters under `PaliGemma/img/...`
- action-side projection heads
- PI05 timestep MLPs

This confirms that the current setup is broader than pure LoRA-only optimization.

## Influence / Data Selection Scripts Built on Top of LoRA Fine-tuning

Several downstream scripts in this repo assume the OpenPI05 LoRA setup and reuse `trainable_filter` to decide which parameters participate in analysis:

- [`scripts/influence_pi05_lora.py`](../scripts/influence_pi05_lora.py)
- [`scripts/influence_pi05_lora_grass.py`](../scripts/influence_pi05_lora_grass.py)
- [`scripts/influence_pi05_lora_grass_window.py`](../scripts/influence_pi05_lora_grass_window.py)
- [`scripts/influence_pi05_lora_grass_leaf_subset.py`](../scripts/influence_pi05_lora_grass_leaf_subset.py)

This means their "LoRA influence" scores should be interpreted consistently with the training setup above:

- usually not pure LoRA-only
- instead, influence over the current trainable parameter subset

## Recommended Mental Model

For this repo, the safest interpretation is:

- model architecture: OpenPI05
- backbone/adapters: Gemma LoRA on language/action-expert transformer blocks
- optimization scope: LoRA plus any parameters not frozen by the current `freeze_filter`

If you need stricter terminology in experiments, it is better to call the current setup:

- "OpenPI05 low-memory fine-tuning with Gemma LoRA adapters"

rather than:

- "strict LoRA-only fine-tuning"

## Files to Read

If you want to trace the implementation end to end, read these files in order:

1. [`src/openpi/models/lora.py`](../src/openpi/models/lora.py)
2. [`src/openpi/models/gemma.py`](../src/openpi/models/gemma.py)
3. [`src/openpi/models/pi0_config.py`](../src/openpi/models/pi0_config.py)
4. [`src/openpi/models/pi0.py`](../src/openpi/models/pi0.py)
5. [`src/openpi/training/config.py`](../src/openpi/training/config.py)
6. [`src/openpi/training/weight_loaders.py`](../src/openpi/training/weight_loaders.py)
7. [`scripts/train.py`](../scripts/train.py)
