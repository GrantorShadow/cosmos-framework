# SABER Stream 2 experiment log

Append-only record of Cosmos3-Nano policy experiments on SABER Stream 2. Every
entry identifies the immutable base checkpoint, training topology, meaningful
configuration differences, W&B run, output checkpoint, and result. Do not put
credentials in this file.

## Shared data and model contract

- Dataset: `DreamVu/SABER-10K`, `SABER-stream2`
- Input: one ego-view video, 33 frames sampled at approximately 15 FPS
- Policy target: 32 future absolute Unitree G1 actions
- State/action width: 72
- Normalization: per-channel mean/std from `meta/stats.json`
- Base: `nvidia/Cosmos3-Nano`, v2 midtrain iteration 6000
- Base revision: `411f42a8fdfb8c5b2583cb8786e0938f49796eaa`
- Fresh modules: `action2llm`, `llm2action`, and `action_modality_embed`

## Runs

### 2026-07-21 — one-H100 LoRA debug, 200 steps

- Purpose: integration and stability smoke test
- W&B: <https://wandb.ai/tristar-ai/cosmos3_saber/runs/6lsm0cnp>
- Topology: 1× H100, batch 1, gradient accumulation 1
- Training: rank-16 LoRA plus fresh 72-D action heads
- Scheduler: production 500-step warmup; all 200 debug steps remained in warmup
- Result: completed; peak allocated GPU memory 38.59 GB
- Loss quarter means:
  - total: 36.128 → 16.884
  - action flow: 3.406 → 1.509
  - vision flow: 0.207 → 0.179
- Checkpoint: `/data/cosmos-runs/cosmos3_saber/modal_debug/saber-g1-200iter/checkpoints/iter_000000200`
- Caveat: not comparable to the production non-LoRA optimizer or batch

### 2026-07-21 — one-H100 non-LoRA memory gate, 2 steps

- Purpose: verify that the reference generation/action parameter selection can
  update on one 80 GB H100 with activation recomputation
- W&B: <https://wandb.ai/tristar-ai/cosmos3_saber/runs/i1ydsj49>
- Topology: 1× H100, batch 1, gradient accumulation 1
- Training: 6,984,498,624 generation/action parameters; no LoRA; EMA disabled
- Optimizer: fused PyTorch AdamW without the production FP32 master copy
- Result: completed; finite losses 47.881 and 21.001
- Checkpoint: `/data/cosmos-runs/cosmos3_saber/modal_debug/saber-g1-full-recompute-2step/checkpoints/iter_000000002`

### 2026-07-21 — one-H100 non-LoRA curve, 200 steps

- Purpose: measure short-run optimization with full-block activation recomputation
- W&B: <https://wandb.ai/tristar-ai/cosmos3_saber/runs/7l32l2zp>
- Modal: <https://modal.com/apps/shauwnakjoshi/main/ap-MPBKHH8PbogOl9bRI0bRcf>
- Topology: 1× H100, batch 1, gradient accumulation 1
- Training: 6,984,498,624 generation/action parameters; no LoRA; EMA disabled
- Optimizer: fused PyTorch AdamW
- Scheduler: 20-step warmup and linear decay through step 200
- Result: completed; peak allocated GPU memory 69.13 GB, NVML usage 71.89 GB
- Loss quarter means:
  - total: 25.153 → 11.725
  - action flow: 2.315 → 0.997
  - vision flow: 0.200 → 0.176
- Checkpoint: `/data/cosmos-runs/cosmos3_saber/modal_debug/saber-g1-full-recompute-200iter/checkpoints/iter_000000200`
- Caveat: training-only result; no held-out evaluation

### 2026-07-21 — eight-H100 production preflight, 2 steps

- Status: pending
- Purpose: exercise the exact production topology and memory path before the
  long run
- Planned topology: 8× H100, FSDP shard degree 8, replicate degree 1
- Planned effective batch: 32 samples/rank × 8 ranks × accumulation 1 = 256
- Planned training: no LoRA; generation/action parameters; full-block
  activation recomputation; EMA enabled
- Planned optimizer: FusedAdam, FP32 master weights, base LR `5e-5`, action-head
  LR `2.5e-4`
- Planned scheduler: production 500-step warmup, 5,000-step linear cycle
- Monitoring-only overrides: loss every step and device memory at step 2
- Known limit: tokenizer AOT compilation is enabled but its normal step-3
  trigger is not reached by a two-step gate
- Planned checkpoint: step 2
- W&B: pending
- Result: pending
