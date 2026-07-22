# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Prepare and test SABER Stream 2 action-policy training on Modal H100s.

Run from the repository root after creating the ``cosmos-saber-secrets``
Modal Secret (required ``HF_TOKEN`` plus optional ``WANDB_API_KEY`` and
``WANDB_ENTITY``)::

    modal run examples/modal_saber_g1.py --action prepare
    modal run examples/modal_saber_g1.py --action train --iterations 2
    modal run --detach examples/modal_saber_g1.py --action train --iterations 20
    modal run --detach examples/modal_saber_g1.py --action train \
        --train-mode full --iterations 200 --require-wandb
    modal run examples/modal_saber_g1.py --action validate-production --iterations 2
    modal run examples/modal_saber_g1.py --action train-production \
        --iterations 2 --require-wandb
"""

from __future__ import annotations

import json
import os
import subprocess
from fractions import Fraction
from pathlib import Path

import modal

APP_NAME = "cosmos-saber-g1"
REPO_URL = "https://github.com/GrantorShadow/cosmos-framework.git"
REPO_COMMIT = "634d91310676c05fed5edfca973bfc9016248210"
REMOTE_REPO = Path("/workspace/cosmos-framework")
DATA_ROOT = Path("/data")
SABER_ROOT = DATA_ROOT / "SABER-10K" / "SABER-stream2"
VAE_PATH = DATA_ROOT / "checkpoints" / "wan22_vae" / "Wan2.2_VAE.pth"
BASE_CHECKPOINT_PATH = DATA_ROOT / "checkpoints" / "Cosmos3-Nano"
OUTPUT_ROOT = DATA_ROOT / "cosmos-runs"
NANO_MIDTRAIN_REPOSITORY = "nvidia/Cosmos3-Nano"
NANO_MIDTRAIN_REVISION = "411f42a8fdfb8c5b2583cb8786e0938f49796eaa"
NANO_MIDTRAIN_EXPERIMENT = "cosmos3_ga_16bm8b_v2_midtrain"
NANO_MIDTRAIN_ITERATION = 6000
NANO_MIDTRAIN_SNAPSHOT = (
    DATA_ROOT
    / "hf-cache"
    / "hub"
    / "models--nvidia--Cosmos3-Nano"
    / "snapshots"
    / NANO_MIDTRAIN_REVISION
)
NANO_MIDTRAIN_PROVENANCE = BASE_CHECKPOINT_PATH / "source.json"

data_volume = modal.Volume.from_name("cosmos-saber-data", create_if_missing=True)
training_secret = modal.Secret.from_name("cosmos-saber-secrets")

image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:25.11-py3")
    .apt_install("curl", "ffmpeg", "git", "git-lfs", "libgl1", "libglib2.0-0", "libx11-dev", "wget")
    .run_commands("python -m pip install --no-cache-dir --upgrade uv==0.11.28")
    .env(
        {
            "GIT_LFS_SKIP_SMUDGE": "1",
            "HF_HOME": str(DATA_ROOT / "hf-cache"),
            "PYTHONUNBUFFERED": "1",
            "UV_LINK_MODE": "copy",
        }
    )
    .run_commands(
        f"git clone --filter=blob:none {REPO_URL} {REMOTE_REPO}",
        f"git -C {REMOTE_REPO} checkout {REPO_COMMIT}",
    )
    .workdir(str(REMOTE_REPO))
    .run_commands("uv sync --frozen --all-extras --group=cu130-train")
)

app = modal.App(APP_NAME, image=image)


def _runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = f"{REMOTE_REPO}/.venv/bin:{env['PATH']}"
    env["PYTHONPATH"] = str(REMOTE_REPO)
    env["LD_LIBRARY_PATH"] = ""
    return env


def _run(command: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=REMOTE_REPO, env=env, check=True)


def _verify_nano_midtrain_provenance(*, record: bool) -> None:
    """Verify that the base DCP came from the pinned public v2 midtrain snapshot."""
    if not NANO_MIDTRAIN_SNAPSHOT.is_dir():
        raise FileNotFoundError(
            "Pinned Cosmos3-Nano midtrain snapshot is missing from the Modal Volume: "
            f"{NANO_MIDTRAIN_SNAPSHOT}"
        )

    expected = {
        "repository": NANO_MIDTRAIN_REPOSITORY,
        "revision": NANO_MIDTRAIN_REVISION,
        "experiment": NANO_MIDTRAIN_EXPERIMENT,
        "iteration": NANO_MIDTRAIN_ITERATION,
    }
    if NANO_MIDTRAIN_PROVENANCE.exists():
        actual = json.loads(NANO_MIDTRAIN_PROVENANCE.read_text())
        if actual != expected:
            raise RuntimeError(
                f"Cosmos3-Nano DCP provenance mismatch: expected {expected}, found {actual}"
            )
    elif record:
        NANO_MIDTRAIN_PROVENANCE.write_text(json.dumps(expected, indent=2) + "\n")
    else:
        raise FileNotFoundError(
            f"Cosmos3-Nano provenance marker is missing: {NANO_MIDTRAIN_PROVENANCE}. "
            "Run --action prepare to verify and record the pinned base."
        )

    print(
        "Verified Cosmos3-Nano base: "
        f"{NANO_MIDTRAIN_EXPERIMENT} iter {NANO_MIDTRAIN_ITERATION}, "
        f"HF revision {NANO_MIDTRAIN_REVISION}"
    )


def _base_training_env(*, nproc_per_node: int) -> dict[str, str]:
    required_paths = [
        SABER_ROOT / "meta" / "info.json",
        SABER_ROOT / "meta" / "modality.json",
        SABER_ROOT / "meta" / "stats.json",
        VAE_PATH,
        BASE_CHECKPOINT_PATH / "checkpoint.json",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Modal training inputs are missing; run --action prepare first. Missing: " + ", ".join(missing)
        )
    _verify_nano_midtrain_provenance(record=False)

    env = _runtime_env()
    env.update(
        {
            "DATASET_PATH": str(SABER_ROOT),
            "SABER_ROOT": str(SABER_ROOT),
            "BASE_CHECKPOINT_PATH": str(BASE_CHECKPOINT_PATH),
            "WAN_VAE_PATH": str(VAE_PATH),
            "IMAGINAIRE_OUTPUT_ROOT": str(OUTPUT_ROOT),
            "NPROC_PER_NODE": str(nproc_per_node),
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        }
    )
    return env


def _training_env(
    iterations: int,
    chunk_length: int,
    run_name: str,
    train_mode: str = "lora",
) -> dict[str, str]:
    """Compose the intentionally reduced one-H100 debug configuration."""
    env = _base_training_env(nproc_per_node=1)

    duration = chunk_length + 1
    save_iter = max(1, min(50, iterations))
    # Keep short debug runs representative: the production recipe uses a
    # 500-step warmup over 5,000 steps, which would leave a 200-step smoke run
    # entirely inside warmup. Preserve the same 10% warmup ratio and complete
    # one linear-decay cycle over the requested debug run.
    warmup_steps = min(500, max(1, iterations // 10))
    wandb_mode = "online" if env.get("WANDB_API_KEY") else "offline"
    overrides = [
        "model.config.parallelism.data_parallel_shard_degree=1",
        "model.config.parallelism.data_parallel_replicate_degree=1",
        "model.config.activation_checkpointing.mode=full",
        "model.config.ema.enabled=false",
        "dataloader_train.max_samples_per_batch=1",
        "dataloader_train.dataloader.batch_size=1",
        "dataloader_train.dataloader.num_workers=2",
        f"dataloader_train.dataloader.datasets.saber_g1.dataset.chunk_length={chunk_length}",
        f"model.config.tokenizer.encode_exact_durations=[{duration}]",
        "trainer.grad_accum_iter=1",
        "trainer.logging_iter=1",
        f"trainer.max_iter={iterations}",
        f"scheduler.warm_up_steps=[{warmup_steps}]",
        f"scheduler.cycle_lengths=[{iterations}]",
        f"checkpoint.save_iter={save_iter}",
        "job.project=cosmos3_saber",
        "job.group=modal_debug",
        f"job.name={run_name}",
        f"job.wandb_mode={wandb_mode}",
    ]
    if train_mode == "lora":
        overrides.extend(
            [
                "model.config.lora_enabled=true",
                "model.config.lora_rank=16",
                "model.config.lora_additional_trainable_modules="
                "'action2llm,llm2action,action_modality_embed'",
                "optimizer.keys_to_select=[lora_,action2llm,llm2action,action_modality_embed]",
                (
                    "checkpoint.keys_to_skip_loading="
                    "[net_ema.,action2llm,llm2action,action_modality_embed,action_pos_embed,lora_]"
                ),
            ]
        )
    elif train_mode == "full":
        overrides.extend(
            [
                "model.config.lora_enabled=false",
                "optimizer.optimizer_type=AdamW",
                (
                    "optimizer.keys_to_select="
                    "[moe_gen,time_embedder,vae2llm,llm2vae,action2llm,llm2action,action_modality_embed]"
                ),
                (
                    "checkpoint.keys_to_skip_loading="
                    "[net_ema.,action2llm,llm2action,action_modality_embed,action_pos_embed]"
                ),
                "trainer.callbacks.compile_tokenizer.enabled=false",
            ]
        )
    else:
        raise ValueError(f"train_mode must be 'lora' or 'full', got {train_mode!r}")
    env["EXTRA_TAIL_OVERRIDES"] = " ".join(overrides)
    return env


def _production_training_env(iterations: int, run_name: str) -> dict[str, str]:
    """Compose the exact eight-H100 Nano production recipe with a short stop."""
    env = _base_training_env(nproc_per_node=8)
    wandb_mode = "online" if env.get("WANDB_API_KEY") else "offline"
    overrides = [
        "model.config.parallelism.data_parallel_shard_degree=8",
        "model.config.parallelism.data_parallel_replicate_degree=1",
        "model.config.activation_checkpointing.mode=full",
        "model.config.lora_enabled=false",
        "model.config.ema.enabled=true",
        "model.config.tokenizer.encode_exact_durations=[33]",
        "optimizer.optimizer_type=FusedAdam",
        (
            "optimizer.keys_to_select="
            "[moe_gen,time_embedder,vae2llm,llm2vae,action2llm,llm2action,action_modality_embed]"
        ),
        "dataloader_train.max_samples_per_batch=16",
        "trainer.grad_accum_iter=2",
        "trainer.logging_iter=1",
        "trainer.callbacks.device_monitor.every_n=1",
        f"trainer.max_iter={iterations}",
        "trainer.callbacks.compile_tokenizer.enabled=true",
        "scheduler.warm_up_steps=[500]",
        "scheduler.cycle_lengths=[5000]",
        f"checkpoint.save_iter={iterations}",
        (
            "checkpoint.keys_to_skip_loading="
            "[net_ema.,action2llm,llm2action,action_modality_embed,action_pos_embed]"
        ),
        "job.project=cosmos3_saber",
        "job.group=production_preflight",
        f"job.name={run_name}",
        f"job.wandb_mode={wandb_mode}",
    ]
    env["EXTRA_TAIL_OVERRIDES"] = " ".join(overrides)
    return env


@app.function(
    cpu=16.0,
    memory=131072,
    timeout=24 * 60 * 60,
    volumes={DATA_ROOT: data_volume},
    secrets=[training_secret],
)
def prepare() -> None:
    """Download Stream 2 and the VAE, then convert Cosmos3-Nano to DCP."""
    env = _runtime_env()

    dataset_marker = DATA_ROOT / ".saber_stream2_ready"
    if not dataset_marker.exists():
        _run(
            [
                "hf",
                "download",
                "DreamVu/SABER-10K",
                "--repo-type",
                "dataset",
                "--include",
                "SABER-stream2/**",
                "--local-dir",
                str(DATA_ROOT / "SABER-10K"),
            ],
            env,
        )
        dataset_marker.touch()
    else:
        print(f"SABER Stream 2 already prepared at {SABER_ROOT}")

    if not VAE_PATH.exists():
        _run(
            [
                "hf",
                "download",
                "Wan-AI/Wan2.2-TI2V-5B",
                "Wan2.2_VAE.pth",
                "--local-dir",
                str(VAE_PATH.parent),
            ],
            env,
        )
    else:
        print(f"Wan2.2 VAE already prepared at {VAE_PATH}")

    if not (BASE_CHECKPOINT_PATH / "checkpoint.json").exists():
        BASE_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
        _run(
            [
                "hf",
                "download",
                NANO_MIDTRAIN_REPOSITORY,
                "--revision",
                NANO_MIDTRAIN_REVISION,
            ],
            env,
        )
        _run(
            [
                "python",
                "-m",
                "cosmos_framework.scripts.convert_model_to_dcp",
                "--checkpoint-path",
                str(NANO_MIDTRAIN_SNAPSHOT),
                "-o",
                str(BASE_CHECKPOINT_PATH),
            ],
            env,
        )
    else:
        print(f"Cosmos3-Nano DCP already prepared at {BASE_CHECKPOINT_PATH}")

    _verify_nano_midtrain_provenance(record=True)

    data_volume.commit()
    print("Preparation complete and committed to Modal Volume cosmos-saber-data.")


@app.function(
    cpu=8.0,
    memory=16384,
    timeout=30 * 60,
    volumes={DATA_ROOT: data_volume},
)
def inspect_saber_videos() -> None:
    """Report SABER episodes whose parquet timestamps exceed their MP4s."""
    import pyarrow.parquet as pq

    meta_root = SABER_ROOT / "meta"
    info = json.loads((meta_root / "info.json").read_text())
    episodes = [json.loads(line) for line in (meta_root / "episodes.jsonl").read_text().splitlines()]
    problems: list[dict[str, object]] = []
    for episode in episodes:
        episode_id = int(episode["episode_index"])
        chunk_index = episode_id // int(info.get("chunks_size", 1000))
        format_args = {
            "episode_chunk": chunk_index,
            "episode_index": episode_id,
            "chunk_index": chunk_index,
            "file_index": episode_id,
            "video_key": "observation.images.ego_view",
        }
        video_path = SABER_ROOT / info["video_path"].format(**format_args)
        data_path = SABER_ROOT / info["data_path"].format(**format_args)
        probe = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=avg_frame_rate,nb_frames,duration",
                    "-of",
                    "json",
                    str(video_path),
                ],
                text=True,
            )
        )["streams"][0]
        video_frames = int(probe["nb_frames"])
        video_fps = float(Fraction(probe["avg_frame_rate"]))
        timestamps = pq.read_table(data_path, columns=["timestamp"])["timestamp"].to_numpy()
        max_requested_frame = int(round(float(timestamps[-1]) * video_fps))
        if len(timestamps) != video_frames or max_requested_frame >= video_frames:
            problems.append(
                {
                    "episode": episode_id,
                    "metadata_length": int(episode["length"]),
                    "parquet_rows": len(timestamps),
                    "video_frames": video_frames,
                    "video_fps": video_fps,
                    "timestamp_min": float(timestamps[0]),
                    "timestamp_max": float(timestamps[-1]),
                    "max_requested_frame": max_requested_frame,
                }
            )

    print(json.dumps(problems, indent=2))
    print(f"SABER video scan: episodes={len(episodes)}, mismatches={len(problems)}")


@app.function(cpu=4.0, memory=16384, timeout=15 * 60)
def test_saber_adapter() -> None:
    """Run the focused SABER adapter tests in the training image."""
    _run(
        [
            "python",
            "-m",
            "pytest",
            "-q",
            "cosmos_framework/data/generator/action/datasets/saber_g1_lerobot_dataset_test.py",
        ],
        _runtime_env(),
    )


@app.function(
    cpu=2.0,
    memory=4096,
    timeout=15 * 60,
    secrets=[training_secret],
)
def inspect_wandb(run_id: str = "6lsm0cnp") -> None:
    """Summarize loss, gradient, and learning-rate histories for a W&B run."""
    _run(
        ["python", "examples/inspect_wandb_run.py", "--run-id", run_id],
        _runtime_env(),
    )


@app.function(
    cpu=2.0,
    memory=4096,
    timeout=15 * 60,
    volumes={DATA_ROOT: data_volume},
)
def audit_full_finetune_params() -> None:
    """Count parameters selected by the reference non-LoRA action recipe."""
    _run(
        ["python", "examples/audit_dcp_params.py", str(BASE_CHECKPOINT_PATH / "model")],
        _runtime_env(),
    )


@app.function(
    cpu=4.0,
    memory=16384,
    timeout=30 * 60,
    volumes={DATA_ROOT: data_volume},
    secrets=[training_secret],
)
def validate(
    iterations: int = 2,
    chunk_length: int = 32,
    run_name: str = "saber-g1-config-check",
    train_mode: str = "lora",
) -> None:
    """Compose the exact debug config on CPU without allocating a GPU."""
    env = _training_env(iterations, chunk_length, run_name, train_mode)
    overrides = env["EXTRA_TAIL_OVERRIDES"].split()
    _run(
        [
            "python",
            "-m",
            "cosmos_framework.scripts.train",
            "--sft-toml",
            "examples/toml/sft_config/action_policy_saber_g1.toml",
            "--dryrun",
            "--",
            *overrides,
        ],
        env,
    )
    print("SABER Stream 2 debug config validated successfully.")


@app.function(
    cpu=4.0,
    memory=16384,
    timeout=30 * 60,
    volumes={DATA_ROOT: data_volume},
    secrets=[training_secret],
)
def validate_production_preflight(
    iterations: int = 2,
    run_name: str = "saber-g1-nano-8xh100-production-2step-ga2",
) -> None:
    """Compose the exact eight-H100 production preflight without allocating GPUs."""
    env = _production_training_env(iterations, run_name)
    overrides = env["EXTRA_TAIL_OVERRIDES"].split()
    _run(
        [
            "python",
            "-m",
            "cosmos_framework.scripts.train",
            "--sft-toml",
            "examples/toml/sft_config/action_policy_saber_g1.toml",
            "--dryrun",
            "--",
            *overrides,
        ],
        env,
    )
    print("SABER Stream 2 eight-H100 production preflight config validated successfully.")


@app.function(
    gpu="H100!",
    cpu=16.0,
    memory=131072,
    timeout=24 * 60 * 60,
    volumes={DATA_ROOT: data_volume},
    secrets=[training_secret],
)
def train(
    iterations: int = 20,
    chunk_length: int = 32,
    run_name: str = "",
    require_wandb: bool = False,
    train_mode: str = "lora",
) -> None:
    """Run SABER with either LoRA or reference generation-expert training."""
    if iterations <= 0:
        raise ValueError(f"iterations must be positive, got {iterations}")
    if chunk_length not in {16, 32}:
        raise ValueError(f"chunk_length must be 16 or 32 for this debug recipe, got {chunk_length}")

    if train_mode not in {"lora", "full"}:
        raise ValueError(f"train_mode must be 'lora' or 'full', got {train_mode!r}")

    run_name = run_name or f"saber_g1_{train_mode}_modal_h100_{iterations}iter"
    env = _training_env(iterations, chunk_length, run_name, train_mode)
    wandb_mode = "online" if env.get("WANDB_API_KEY") else "offline"
    if require_wandb and wandb_mode != "online":
        raise RuntimeError(
            "Online W&B logging was required, but WANDB_API_KEY is missing from "
            "the cosmos-saber-secrets Modal Secret."
        )

    print(f"Starting {iterations}-iteration Modal H100 {train_mode} debug run: {run_name}")
    print(f"W&B mode: {wandb_mode}")
    if train_mode == "full":
        print("No LoRA; full-block activation recomputation; reduced-memory fused PyTorch AdamW.")
    else:
        print("This is a LoRA integration/stability run, not the final 8-H100 full-finetune recipe.")
    _run(["bash", "examples/launch_sft_action_policy_saber_g1.sh"], env)
    data_volume.commit()
    print(f"Training outputs committed under {OUTPUT_ROOT}/cosmos3_saber/modal_debug/{run_name}")


@app.function(
    gpu="H100:8",
    cpu=64.0,
    memory=262144,
    timeout=24 * 60 * 60,
    volumes={DATA_ROOT: data_volume},
    secrets=[training_secret],
)
def train_production_preflight(
    iterations: int = 2,
    run_name: str = "saber-g1-nano-8xh100-production-2step-ga2",
    require_wandb: bool = True,
) -> None:
    """Run a bounded eight-H100 gate with the exact Nano production recipe."""
    if not 1 <= iterations <= 10:
        raise ValueError(f"production preflight iterations must be in [1, 10], got {iterations}")

    env = _production_training_env(iterations, run_name)
    wandb_mode = "online" if env.get("WANDB_API_KEY") else "offline"
    if require_wandb and wandb_mode != "online":
        raise RuntimeError(
            "Online W&B logging was required, but WANDB_API_KEY is missing from "
            "the cosmos-saber-secrets Modal Secret."
        )

    print(f"Starting {iterations}-iteration eight-H100 Nano production preflight: {run_name}")
    print(f"W&B mode: {wandb_mode}")
    print(
        "No LoRA; FSDP shard degree 8; FusedAdam with FP32 masters; EMA enabled; "
        "full-block activation recomputation; 16 samples/rank with accumulation 2 "
        "(effective global batch 256)."
    )
    _run(["bash", "examples/launch_sft_action_policy_saber_g1.sh"], env)
    data_volume.commit()
    print(f"Training outputs committed under {OUTPUT_ROOT}/cosmos3_saber/production_preflight/{run_name}")


@app.local_entrypoint()
def main(
    action: str = "train",
    iterations: int = 20,
    chunk_length: int = 32,
    run_name: str = "",
    require_wandb: bool = False,
    train_mode: str = "lora",
    wandb_run_id: str = "6lsm0cnp",
) -> None:
    """Dispatch preparation, one-H100 debugging, or the eight-H100 preflight."""
    if action == "prepare":
        prepare.remote()
    elif action == "inspect":
        inspect_saber_videos.remote()
    elif action == "test":
        test_saber_adapter.remote()
    elif action == "inspect-wandb":
        inspect_wandb.remote(run_id=wandb_run_id)
    elif action == "audit-full":
        audit_full_finetune_params.remote()
    elif action == "validate":
        validate.remote(
            iterations=iterations,
            chunk_length=chunk_length,
            run_name=run_name,
            train_mode=train_mode,
        )
    elif action == "validate-production":
        validate_production_preflight.remote(
            iterations=iterations,
            run_name=run_name or "saber-g1-nano-8xh100-production-2step-ga2",
        )
    elif action == "train":
        train.remote(
            iterations=iterations,
            chunk_length=chunk_length,
            run_name=run_name,
            require_wandb=require_wandb,
            train_mode=train_mode,
        )
    elif action == "train-production":
        train_production_preflight.remote(
            iterations=iterations,
            run_name=run_name or "saber-g1-nano-8xh100-production-2step-ga2",
            require_wandb=require_wandb,
        )
    else:
        raise ValueError(
            "action must be 'prepare', 'inspect', 'test', 'inspect-wandb', 'audit-full', "
            "'validate', 'validate-production', 'train', or 'train-production', "
            f"got {action!r}"
        )
