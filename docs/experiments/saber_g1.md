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

- Status: failed during step 2 backward; no checkpoint written
- Purpose: exercise the exact production topology and memory path before the
  long run
- Training commit: `403db599c885a938f3f1ef7c3bddf164c5a25555`
- Modal: <https://modal.com/apps/shauwnakjoshi/main/ap-I7x4sSkXh7aoT0o2KAxuaX>
- W&B: <https://wandb.ai/tristar-ai/cosmos3_saber/runs/a0bw3tvp>
- Topology: 8× H100, FSDP shard degree 8, replicate degree 1
- Effective batch: 32 samples/rank × 8 ranks × accumulation 1 = 256
- Training: no LoRA; generation/action parameters; full-block
  activation recomputation; EMA enabled
- Optimizer: FusedAdam, FP32 master weights, base LR `5e-5`, action-head
  LR `2.5e-4`
- Scheduler: production 500-step warmup, 5,000-step linear cycle
- Step 1: completed on every rank in 163.4 seconds; finite per-rank loss range
  19.746–30.399; synchronized pre-clip gradient norm 8.5625
- Failure: step 2 recomputed backward requested another 1.47 GiB with only
  0.13–0.68 GiB free per GPU; rank 0 reported 78.72/79.18 GiB in use
- Conclusion: 32 samples/rank does not have safe headroom for variable packed
  sequence lengths on 80 GB H100s
- Known limit: tokenizer AOT compilation is enabled but its normal step-3
  trigger is not reached by a two-step gate
- Checkpoint: none

### 2026-07-21 — eight-H100 production preflight retry, 2 steps

- Status: completed
- Purpose: preserve effective global batch 256 while reducing peak activation
  memory after the 32×1 OOM
- Training commit: `634d91310676c05fed5edfca973bfc9016248210`
- Modal runner commit: `3f0b2f4dce82b7b0130168b9f0bbd0b4f25db5a5`
- Modal: <https://modal.com/apps/shauwnakjoshi/main/ap-s1dG1XMUajq4AsrfswL9Lu>
- W&B: <https://wandb.ai/tristar-ai/cosmos3_saber/runs/w04pgxz9>
- Topology: 8× H100, FSDP shard degree 8, replicate degree 1
- Effective batch: 16 samples/rank × 8 ranks × accumulation 2 = 256
- Training: no LoRA; 6,984,498,624 generation/action parameters; full-block
  activation recomputation; EMA enabled
- Optimizer: FusedAdam, FP32 master weights, base LR `5e-5`, action-head
  LR `2.5e-4`
- Scheduler: production 500-step warmup, 5,000-step linear cycle
- Monitoring-only overrides: loss and device memory every step
- W&B `train/loss`: 22.5915 → 22.5549
- Per-rank loss range: 19.524–30.066 at step 1; 19.522–31.259 at step 2
- Synchronized pre-clip gradient norm: 5.2500 → 4.84375
- Base learning rate during warmup: `5.0e-11` → `1.0005e-7`
- Peak memory at step 2: 54.92 GiB PyTorch allocated and 65.42 GiB maximum
  NVML usage; minimum free memory across ranks 14.22 GiB
- Tokenizer: all 20 AOT variants compiled and loaded across eight ranks in
  85.6 seconds
- Checkpoint save: completed in 61.2 seconds
- Checkpoint:
  `/data/cosmos-runs/cosmos3_saber/production_preflight/saber-g1-nano-8xh100-production-2step-ga2/checkpoints/iter_000000002`
- Conclusion: the 16×2 configuration preserves global batch 256 with safe
  memory headroom and is the production baseline for the long run
- Caveat: this validates distributed mechanics and memory, not convergence or
  held-out policy quality

### 2026-07-21 — four-H100 Nano run, 50 iterations

- Status: planned
- Purpose: run approximately half an epoch on four H100s with production model
  and optimizer behavior
- Base: `nvidia/Cosmos3-Nano`, v2 midtrain iteration 6000, revision
  `411f42a8fdfb8c5b2583cb8786e0938f49796eaa`
- Topology: 4× H100, FSDP shard degree 4, replicate degree 1
- Effective batch: 4 samples/rank × 4 ranks × accumulation 16 = 256
- Duration: 50 optimizer iterations, approximately 0.52 training epochs
- Training: no LoRA; 6,984,498,624 generation/action parameters; bfloat16;
  full-block activation recomputation; EMA enabled
- Optimizer: FusedAdam with FP32 master weights, weight decay `0.05`, betas
  `[0.9, 0.99]`, epsilon `1e-8`, base LR `5e-5`, action-head LR `2.5e-4`
- Scheduler: LambdaLinear, 500-step warmup, 5,000-step cycle, factors
  `1e-6 → 1.0 → 0.0`
- Dataset: 174 train episodes, 24,718 windows, 33 frames at 15 FPS, 32 future
  actions, 72 channels, per-channel mean/std normalization
- Logging: W&B every optimizer step, device memory every optimizer step, raw
  launcher stdout/stderr, resolved config, and non-secret JSON run manifest
- Checkpoints: iterations 25 and 50
- Output: `/data/cosmos-runs/cosmos3_saber/four_h100_train/saber-g1-nano-4xh100-50iter`
- W&B: pending
- Modal: pending
- Result: pending
