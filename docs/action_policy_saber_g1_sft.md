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
[Training](./training.md#step-2--prepare-checkpoint), then export:

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

