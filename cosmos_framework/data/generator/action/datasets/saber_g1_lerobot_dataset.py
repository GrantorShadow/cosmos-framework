# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""DreamVu SABER Stream 2 (Unitree G1) LeRobot-v2 dataset adapter.

The public ``SABER-stream2`` release stores one parquet and one ego-view MP4
per episode.  Its state/action contract is an identical 72-channel absolute
G1 configuration.  At row ``t``, ``action[t]`` is the target state for
``t + 1``.  This adapter therefore emits:

    [observation.state[t], action[t], ..., action[t + T - 1]]

The leading state is consumed by ``ActionTransformPipeline`` as a clean action
conditioning token; the remaining entries are flow-matching targets.
"""

from __future__ import annotations

import json
import random
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from cosmos_framework.data.generator.action.action_processing import (
    ActionNormalizer,
    QuaternionRenormalizingActionNormalizer,
    load_action_stats,
    resolve_action_normalization,
)
from cosmos_framework.data.generator.action.domain_utils import get_domain_id
from cosmos_framework.utils import log

_MODE_CHOICES = ("forward_dynamics", "inverse_dynamics", "policy")
_STATE_KEY = "observation.state"
_ACTION_KEY = "action"
_VIDEO_KEY = "observation.images.ego_view"
_ACTION_DIM = 72
_ROOT_QUATERNION_SLICE = slice(3, 7)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Required SABER metadata file not found: {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _resolve_stream2_root(root: str | Path) -> Path:
    """Accept either the stream directory or its SABER-10K parent."""
    candidate = Path(root).expanduser()
    if (candidate / "meta" / "info.json").exists():
        return candidate
    nested = candidate / "SABER-stream2"
    if (nested / "meta" / "info.json").exists():
        return nested
    raise FileNotFoundError(
        f"SABER Stream 2 root must contain meta/info.json either directly or under SABER-stream2; got {candidate}"
    )


class SABERG1LeRobotDataset(Dataset):
    """Map-style policy windows over the released SABER Stream 2 dataset."""

    def __init__(
        self,
        root: str,
        fps: float = 15.0,
        chunk_length: int = 32,
        mode: str = "policy",
        split: str = "train",
        val_ratio: float = 0.05,
        seed: int = 42,
        action_normalization: str | None = "meanstd",
        sample_stride: int = 1,
        tolerance_s: float = 1e-4,
        max_cached_episodes: int = 8,
    ) -> None:
        super().__init__()
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        if chunk_length <= 0:
            raise ValueError(f"chunk_length must be positive, got {chunk_length}")
        if sample_stride <= 0:
            raise ValueError(f"sample_stride must be positive, got {sample_stride}")
        if mode not in (*_MODE_CHOICES, "joint"):
            raise ValueError(f"Unsupported mode={mode!r}; use one of {_MODE_CHOICES} or 'joint'.")
        if action_normalization not in (None, "meanstd", "minmax"):
            raise ValueError(
                "SABER's supplied stats contain mean/std/min/max but no q01/q99; "
                "use action_normalization='meanstd', 'minmax', or None."
            )
        split = split.lower().strip()
        if split not in {"train", "val", "valid", "validation", "eval", "test", "full"}:
            raise ValueError(f"Unsupported split={split!r}; use train/val/full.")
        if split != "full" and not (0.0 < val_ratio < 1.0):
            raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
        if max_cached_episodes <= 0:
            raise ValueError(f"max_cached_episodes must be positive, got {max_cached_episodes}")

        self._root = _resolve_stream2_root(root)
        self._mode = mode
        self._chunk_length = int(chunk_length)
        self._sample_stride = int(sample_stride)
        self._tolerance_s = float(tolerance_s)
        self._max_cached_episodes = int(max_cached_episodes)
        self._domain_id = get_domain_id("saber_g1")

        meta_root = self._root / "meta"
        self._info = json.loads((meta_root / "info.json").read_text())
        self._modality = json.loads((meta_root / "modality.json").read_text())
        self._validate_metadata()

        self._native_fps = float(self._info["fps"])
        self._frame_stride = max(1, int(round(self._native_fps / float(fps))))
        self._fps = self._native_fps / self._frame_stride
        if not np.isclose(self._fps, fps, rtol=0.02, atol=0.02):
            raise ValueError(
                f"Requested fps={fps} cannot be represented by integer subsampling of "
                f"native fps={self._native_fps}; nearest supported fps is {self._fps:.6f}."
            )

        episodes = {int(row["episode_index"]): row for row in _read_jsonl(meta_root / "episodes.jsonl")}
        self._tasks = {int(row["task_index"]): str(row["task"]) for row in _read_jsonl(meta_root / "tasks.jsonl")}
        selected_episode_ids = self._split_episode_ids(sorted(episodes), split, val_ratio, seed)
        self._episodes = {episode_id: episodes[episode_id] for episode_id in selected_episode_ids}

        self._episode_records: list[tuple[int, int]] = []
        self._episode_cum_ends: list[int] = []
        total = 0
        raw_window_span = self._chunk_length * self._frame_stride
        for episode_id in selected_episode_ids:
            episode_length = int(episodes[episode_id]["length"])
            raw_valid_starts = max(0, episode_length - raw_window_span)
            valid_starts = (raw_valid_starts + self._sample_stride - 1) // self._sample_stride
            if valid_starts == 0:
                continue
            self._episode_records.append((episode_id, valid_starts))
            total += valid_starts
            self._episode_cum_ends.append(total)

        self._episode_cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self._action_normalizer = self._build_action_normalizer(action_normalization)

        log.info(
            f"Loaded SABER G1 root={self._root} split={split!r} "
            f"native_fps={self._native_fps:.5g} sampled_fps={self._fps:.5g} "
            f"frame_stride={self._frame_stride} episodes={len(self._episode_records)} "
            f"valid_windows={len(self)}"
        )

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def chunk_length(self) -> int:
        return self._chunk_length

    @property
    def action_dim(self) -> int:
        return _ACTION_DIM

    @property
    def action_names(self) -> list[str]:
        return list(self._info["features"][_ACTION_KEY]["names"])

    def get_action_normalizer(self, _sample: dict[str, Any] | None = None) -> ActionNormalizer | None:
        """Return the normalizer consumed by ``ActionTransformPipeline``."""
        return self._action_normalizer

    def __len__(self) -> int:
        return self._episode_cum_ends[-1] if self._episode_cum_ends else 0

    def get_shuffle_blocks(self) -> list[tuple[int, int]]:
        """Return per-episode flat-index blocks for iterable episode shuffling."""
        blocks: list[tuple[int, int]] = []
        previous = 0
        for end in self._episode_cum_ends:
            blocks.append((previous, end - previous))
            previous = end
        return blocks

    def __getitem__(self, idx: int) -> dict[str, Any]:
        episode_id, start = self._resolve_index(int(idx))
        episode = self._load_episode(episode_id)

        video_indices = start + np.arange(self._chunk_length + 1, dtype=np.int64) * self._frame_stride
        # SABER action[t] equals the target observation.state[t + 1].  When
        # video is temporally subsampled, use the action immediately preceding
        # each selected future video frame.
        action_indices = video_indices[1:] - 1

        initial_state = torch.from_numpy(episode["state"][video_indices[0]].copy()).float()  # [72]
        future_actions = torch.from_numpy(episode["action"][action_indices].copy()).float()  # [T,72]
        action = torch.cat([initial_state.unsqueeze(0), future_actions], dim=0)  # [T+1,72]

        timestamps = episode["timestamp"][video_indices]
        video = self._decode_video(episode_id, timestamps)  # [T+1,C,H,W], float [0,1] or uint8
        if video.ndim != 4:
            raise ValueError(f"Decoded SABER video must have shape [T,C,H,W], got {tuple(video.shape)}")
        if torch.is_floating_point(video):
            video = (video * 255.0).clamp(0.0, 255.0).to(torch.uint8)
        elif video.dtype != torch.uint8:
            raise TypeError(f"Decoded SABER video must be floating point or uint8, got {video.dtype}")
        video = video.permute(1, 0, 2, 3).contiguous()  # [C,T+1,H,W]

        task_index = int(episode["task_index"][video_indices[0]])
        if task_index not in self._tasks:
            raise KeyError(f"SABER task_index={task_index} is missing from meta/tasks.jsonl")

        mode = random.choice(_MODE_CHOICES) if self._mode == "joint" else self._mode
        return {
            "ai_caption": self._tasks[task_index],
            "video": video,
            "action": action,
            "conditioning_fps": torch.tensor(self._fps, dtype=torch.float32),
            "mode": mode,
            "domain_id": torch.tensor(self._domain_id, dtype=torch.long),
            "viewpoint": "ego_view",
        }

    def _validate_metadata(self) -> None:
        features = self._info.get("features", {})
        for key in (_STATE_KEY, _ACTION_KEY, _VIDEO_KEY):
            if key not in features:
                raise ValueError(f"SABER metadata is missing required feature {key!r}")
        state_feature = features[_STATE_KEY]
        action_feature = features[_ACTION_KEY]
        if state_feature.get("shape") != [_ACTION_DIM] or action_feature.get("shape") != [_ACTION_DIM]:
            raise ValueError(
                f"SABER G1 requires 72-D state/action, got state={state_feature.get('shape')} "
                f"action={action_feature.get('shape')}"
            )
        if state_feature.get("names") != action_feature.get("names"):
            raise ValueError("SABER state and action channel names/order must match exactly.")
        names = action_feature.get("names") or []
        expected_quaternion = ["root_quat_x", "root_quat_y", "root_quat_z", "root_quat_w"]
        if names[_ROOT_QUATERNION_SLICE] != expected_quaternion:
            raise ValueError(
                f"Unexpected SABER root quaternion contract: {names[_ROOT_QUATERNION_SLICE]}; "
                f"expected {expected_quaternion}"
            )

        state_groups = self._modality.get("state", {})
        action_groups = self._modality.get("action", {})
        if state_groups.keys() != action_groups.keys():
            raise ValueError("SABER modality.json state/action groups must match.")
        covered = 0
        for group_name, state_group in state_groups.items():
            action_group = action_groups[group_name]
            for field in ("start", "end", "rotation_type", "absolute", "dtype", "range"):
                if state_group.get(field) != action_group.get(field):
                    raise ValueError(f"SABER state/action modality mismatch for {group_name}.{field}")
            if not state_group.get("absolute", False):
                raise ValueError(f"SABER group {group_name!r} must use absolute coordinates.")
            covered += int(state_group["end"]) - int(state_group["start"])
        if covered != _ACTION_DIM:
            raise ValueError(f"SABER modality groups cover {covered} channels, expected {_ACTION_DIM}.")

    def _build_action_normalizer(self, action_normalization: str | None) -> ActionNormalizer | None:
        if action_normalization is None:
            return None
        raw_stats = load_action_stats(str(self._root / "meta" / "stats.json"), stats_key="action")
        required = {"mean", "std"} if action_normalization == "meanstd" else {"min", "max"}
        missing = required - raw_stats.keys()
        if missing:
            raise ValueError(f"SABER action stats are missing fields required by {action_normalization}: {missing}")
        stats = {key: torch.from_numpy(value).float() for key, value in raw_stats.items()}
        for key in required:
            if stats[key].shape != (_ACTION_DIM,):
                raise ValueError(f"SABER action stat {key!r} must have shape [72], got {tuple(stats[key].shape)}")
        affine = resolve_action_normalization(action_normalization, stats)
        return QuaternionRenormalizingActionNormalizer(
            affine,
            quaternion_start=_ROOT_QUATERNION_SLICE.start,
            quaternion_end=_ROOT_QUATERNION_SLICE.stop,
        )

    @staticmethod
    def _split_episode_ids(
        episode_ids: list[int],
        split: str,
        val_ratio: float,
        seed: int,
    ) -> list[int]:
        if split == "full":
            return episode_ids
        num_val = max(1, int(round(len(episode_ids) * val_ratio)))
        rng = random.Random(seed)
        val_ids = set(rng.sample(episode_ids, num_val))
        if split == "train":
            return [episode_id for episode_id in episode_ids if episode_id not in val_ids]
        return [episode_id for episode_id in episode_ids if episode_id in val_ids]

    def _resolve_index(self, idx: int) -> tuple[int, int]:
        size = len(self)
        if idx < 0:
            idx += size
        if idx < 0 or idx >= size:
            raise IndexError(f"SABER index {idx} out of range for size {size}")
        record_index = bisect_right(self._episode_cum_ends, idx)
        record_start = 0 if record_index == 0 else self._episode_cum_ends[record_index - 1]
        episode_id, _ = self._episode_records[record_index]
        raw_start = (idx - record_start) * self._sample_stride
        return episode_id, raw_start

    def _episode_data_path(self, episode_id: int) -> Path:
        chunks_size = int(self._info.get("chunks_size", 1000))
        chunk_index = episode_id // chunks_size
        relative = self._info["data_path"].format(
            episode_chunk=chunk_index,
            episode_index=episode_id,
            chunk_index=chunk_index,
            file_index=episode_id,
        )
        return self._root / relative

    def _video_path(self, episode_id: int) -> Path:
        chunks_size = int(self._info.get("chunks_size", 1000))
        chunk_index = episode_id // chunks_size
        relative = self._info["video_path"].format(
            episode_chunk=chunk_index,
            episode_index=episode_id,
            chunk_index=chunk_index,
            file_index=episode_id,
            video_key=_VIDEO_KEY,
        )
        return self._root / relative

    def _load_episode(self, episode_id: int) -> dict[str, np.ndarray]:
        cached = self._episode_cache.get(episode_id)
        if cached is not None:
            self._episode_cache.move_to_end(episode_id)
            return cached

        path = self._episode_data_path(episode_id)
        if not path.exists():
            raise FileNotFoundError(f"SABER episode parquet not found: {path}")
        table = pq.read_table(path, columns=[_STATE_KEY, _ACTION_KEY, "timestamp", "task_index"])
        episode = {
            "state": np.asarray(table[_STATE_KEY].to_pylist(), dtype=np.float32),
            "action": np.asarray(table[_ACTION_KEY].to_pylist(), dtype=np.float32),
            "timestamp": table["timestamp"].to_numpy().astype(np.float64, copy=False),
            "task_index": table["task_index"].to_numpy().astype(np.int64, copy=False),
        }
        expected_length = int(self._episodes[episode_id]["length"])
        if episode["state"].shape != (expected_length, _ACTION_DIM):
            raise ValueError(
                f"SABER episode {episode_id} state shape {episode['state'].shape} "
                f"does not match metadata length={expected_length}, dim={_ACTION_DIM}"
            )
        if episode["action"].shape != (expected_length, _ACTION_DIM):
            raise ValueError(
                f"SABER episode {episode_id} action shape {episode['action'].shape} "
                f"does not match metadata length={expected_length}, dim={_ACTION_DIM}"
            )

        self._episode_cache[episode_id] = episode
        while len(self._episode_cache) > self._max_cached_episodes:
            self._episode_cache.popitem(last=False)
        return episode

    def _decode_video(self, episode_id: int, timestamps: np.ndarray) -> torch.Tensor:
        from lerobot.datasets.video_utils import decode_video_frames

        path = self._video_path(episode_id)
        if not path.exists():
            raise FileNotFoundError(f"SABER episode video not found: {path}")
        return decode_video_frames(path, timestamps.tolist(), self._tolerance_s)
