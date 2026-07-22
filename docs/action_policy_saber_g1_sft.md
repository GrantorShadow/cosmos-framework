# Cosmos3-Nano SABER Stream 2 action-policy SFT

This recipe adapts `nvidia/Cosmos3-Nano` to the public
[`DreamVu/SABER-10K`](https://huggingface.co/datasets/DreamVu/SABER-10K)
`SABER-stream2` release: 640×360 ego video, task text, and absolute 72-D
Unitree-G1 state/action targets.

## Model contract

The adapter verifies `meta/info.json` and `meta/modality.json` before loading
any episode. State and action must both be 72-D, have identical names/order,
and use absolute coordinates. Each training window is:

```text
video:  [3, 33, H, W] at ~15 FPS
action: [33, 72]
        ├─ row 0: normalized observation.state (clean conditioning)
        └─ rows 1:33: normalized future absolute action targets
```

SABER stores `action[t] == observation.state[t+1]`. For the default 2× temporal
subsampling, the loader selects the action immediately preceding each sampled
future video frame. `policy` mode conditions vision frame 0 and action row 0;
all remaining video/action entries are flow-matching targets.

The supplied action `mean`/`std` statistics normalize all 72 channels. The
same action normalizer is applied to the leading state so it occupies the same
model space. Generated root quaternion channels `[3:7]` are projected back to
unit length after denormalization.

## Files

| Purpose | File |
| --- | --- |
| LeRobot-v2 adapter | `cosmos_framework/data/generator/action/datasets/saber_g1_lerobot_dataset.py` |
| SFT wrapper | `cosmos_framework/data/generator/action/datasets/action_sft_dataset.py` |
| Model experiment | `cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_saber_g1_nano.py` |
| Run configuration | `examples/toml/sft_config/action_policy_saber_g1.toml` |
| Launcher | `examples/launch_sft_action_policy_saber_g1.sh` |
| Modal H100 debug/production app | `examples/modal_saber_g1.py` |
| Experiment ledger | `docs/experiments/saber_g1.md` |

## Prepare inputs

Accept the gated dataset terms first, authenticate the Hugging Face CLI, and
download only Stream 2:

```shell
hf download DreamVu/SABER-10K \
  --repo-type dataset \
  --include "SABER-stream2/**" \
  --local-dir /data/SABER-10K
```

Convert the public Nano checkpoint to DCP as described in
[Training](./training.md#step-2--prepare-checkpoint), then export. In the Modal
recipe this source is pinned to the public v2 midtraining checkpoint
(`cosmos3_ga_16bm8b_v2_midtrain`, iteration 6000) instead of the mutable
Hugging Face `main` branch:

```shell
export DATASET_PATH=/data/SABER-10K/SABER-stream2
export BASE_CHECKPOINT_PATH=/data/checkpoints/Cosmos3-Nano
export WAN_VAE_PATH=/data/checkpoints/Wan2.2_VAE.pth
export IMAGINAIRE_OUTPUT_ROOT=/data/cosmos-runs
```

## Validate before a full run

Run a short configuration/data smoke test:

```shell
export EXTRA_TAIL_OVERRIDES="trainer.max_iter=2 checkpoint.save_iter=2 \
  dataloader_train.max_samples_per_batch=1 \
  dataloader_train.dataloader.num_workers=1 \
  dataloader_train.dataloader.batch_size=1"
bash examples/launch_sft_action_policy_saber_g1.sh
```

Then remove `EXTRA_TAIL_OVERRIDES` and launch the configured run:

```shell
unset EXTRA_TAIL_OVERRIDES
bash examples/launch_sft_action_policy_saber_g1.sh
```

The starter recipe uses eight-way FSDP, 480p processing, 33-frame windows,
fresh 72-D action projections, a `5e-5` shared learning rate, and a 5× action
head multiplier. These are starting values, not a published SABER/Cosmos
reproduction.

## Debug on one Modal H100

The single-H100 Modal path is an integration and stability check with two
training modes:

- `--train-mode lora` (the default) enables rank-16 LoRA on the generation
  transformer and trains the fresh 72-D action heads.
- `--train-mode full` disables LoRA and trains the reference generation expert
  plus the fresh action heads (about 6.98B parameters). It checkpoints each
  whole transformer block, discarding its intermediate activations in the
  forward pass and recomputing them during backward. EMA and tokenizer AOT
  compilation are disabled. To fit one 80 GB H100 it uses fused PyTorch AdamW;
  the final eight-H100 recipe uses the reference FusedAdam optimizer instead.

Both modes use batch size one and are debug runs, so their training curves do
not establish validation quality or replace the final distributed run.

Install and authenticate Modal locally:

```shell
python3 -m pip install modal
python3 -m modal setup
```

Create a Modal Secret named `cosmos-saber-secrets` containing `HF_TOKEN`.
Add `WANDB_API_KEY` and optionally `WANDB_ENTITY` for online W&B; without the
API key, the runner explicitly logs W&B offline on the persistent Volume. The
dashboard secret editor is recommended so credentials do not enter shell
history. The app creates and uses a persistent Volume named
`cosmos-saber-data`. Its image clones the exact pushed SABER implementation
commit from the public fork instead of uploading the surrounding local working
tree.

Prepare the dataset, VAE, and DCP checkpoint without billing GPU time:

```shell
modal run examples/modal_saber_g1.py --action prepare
```

Run the stages in order:

```shell
# Wiring check: model/data init, forward/backward, W&B, checkpoint.
modal run examples/modal_saber_g1.py --action train --iterations 2

# Short stability check. --detach keeps it alive if the local terminal exits.
modal run --detach examples/modal_saber_g1.py --action train --iterations 20

# Optional: exercises checkpoints at iterations 50 and 100.
modal run --detach examples/modal_saber_g1.py --action train --iterations 100
```

For a no-LoRA activation-recomputation test, first run a two-step memory gate,
then the longer curve only if it succeeds:

```shell
modal run examples/modal_saber_g1.py \
  --action train --train-mode full --iterations 2 --require-wandb

modal run --detach examples/modal_saber_g1.py \
  --action train --train-mode full --iterations 200 --require-wandb
```

The debug runner scales the production schedule to the requested length: a
10% warmup followed by linear decay through the final step. Thus a 200-step
run uses 20 warmup steps instead of spending all 200 steps inside the
production 500-step warmup. If a 33-frame sample exhausts H100 memory, retry
the same stage with `--chunk-length 16`. Modal outputs persist under
`/data/cosmos-runs` on the Volume, and online metrics appear in the
`cosmos3_saber/modal_debug` W&B project/group.

## Preflight on eight Modal H100s

Before a long run, validate the fully resolved production configuration on CPU,
then run the bounded two-step distributed gate:

```shell
modal run examples/modal_saber_g1.py \
  --action validate-production --iterations 2 \
  --run-name saber-g1-nano-8xh100-production-2step

modal run examples/modal_saber_g1.py \
  --action train-production --iterations 2 \
  --run-name saber-g1-nano-8xh100-production-2step --require-wandb
```

This path uses the production topology and memory behavior: eight-way FSDP,
batch 32 per rank (global batch 256), no LoRA, the reference FusedAdam with
FP32 master weights, EMA, full-block activation recomputation, tokenizer AOT
compilation, and the 500/5,000-step scheduler. Only the stop, save, logging,
and device-monitor intervals are shortened for the gate. It starts from the
pinned public Nano midtraining checkpoint rather than resuming a one-H100
debug checkpoint. The normal tokenizer compilation callback begins after step
3, so it is configured but not reached by this two-step gate.

Results and exact run provenance are recorded in the
[SABER Stream 2 experiment ledger](./experiments/saber_g1.md).
