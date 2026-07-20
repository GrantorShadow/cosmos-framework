# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from cosmos_framework.data.generator.action.action_processing import ActionProcessor
from cosmos_framework.data.generator.action.datasets.saber_g1_lerobot_dataset import SABERG1LeRobotDataset
from cosmos_framework.data.generator.action.domain_utils import get_action_dim, get_domain_id
from cosmos_framework.data.generator.action.transforms import build_sequence_plan_from_mode


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _action_names() -> list[str]:
    return [
        "root_pos_x",
        "root_pos_y",
        "root_pos_z",
        "root_quat_x",
        "root_quat_y",
        "root_quat_z",
        "root_quat_w",
        *[f"joint_{i}" for i in range(65)],
    ]


def _make_stream2_root(tmp_path: Path) -> Path:
    root = tmp_path / "SABER-10K" / "SABER-stream2"
    meta = root / "meta"
    data = root / "data" / "chunk-000"
    meta.mkdir(parents=True)
    data.mkdir(parents=True)

    names = _action_names()
    info = {
        "codebase_version": "v2.0",
        "robot_type": "humanoid",
        "total_episodes": 1,
        "total_frames": 5,
        "total_tasks": 1,
        "total_videos": 1,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": 30.0,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.ego_view": {"dtype": "video", "shape": [8, 8, 3]},
            "observation.state": {"dtype": "float32", "shape": [72], "names": names},
            "action": {"dtype": "float32", "shape": [72], "names": names},
        },
    }
    (meta / "info.json").write_text(json.dumps(info))

    group = {
        "start": 0,
        "end": 72,
        "rotation_type": None,
        "absolute": True,
        "dtype": "float32",
        "range": None,
    }
    modality = {
        "state": {"robot_body": {"original_key": "observation.state", **group}},
        "action": {"robot_body": {"original_key": "action", **group}},
        "video": {"ego_view": {"original_key": "observation.images.ego_view"}},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
    (meta / "modality.json").write_text(json.dumps(modality))
    _write_jsonl(meta / "episodes.jsonl", [{"episode_index": 0, "tasks": [0], "length": 5}])
    _write_jsonl(meta / "tasks.jsonl", [{"task_index": 0, "task": "Walk to the produce aisle."}])

    stats = {
        "action": {
            "mean": [0.0] * 72,
            "std": [2.0] * 72,
            "min": [-10.0] * 72,
            "max": [10.0] * 72,
        }
    }
    (meta / "stats.json").write_text(json.dumps(stats))

    state = torch.stack([torch.full((72,), float(i)) for i in range(5)])
    state[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    # The released SABER contract is action[t] == observation.state[t + 1].
    action = torch.cat([state[1:], state[-1:]], dim=0)
    table = pa.table(
        {
            "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32())),
            "action": pa.array(action.tolist(), type=pa.list_(pa.float32())),
            "timestamp": pa.array([i / 30.0 for i in range(5)], type=pa.float64()),
            "task_index": pa.array([0] * 5, type=pa.int64()),
        }
    )
    pq.write_table(table, data / "episode_000000.parquet")
    return root


def _stub_video(_episode_id: int, timestamps) -> torch.Tensor:
    return torch.zeros((len(timestamps), 3, 8, 8), dtype=torch.float32)


def test_saber_stream2_alignment_and_state_conditioning(tmp_path, monkeypatch):
    root = _make_stream2_root(tmp_path)
    dataset = SABERG1LeRobotDataset(
        root=str(root.parent),
        fps=15.0,
        chunk_length=2,
        split="full",
        action_normalization=None,
    )
    monkeypatch.setattr(dataset, "_decode_video", _stub_video)

    sample = dataset[0]

    assert sample["action"].shape == (3, 72)
    # 30 -> 15 FPS selects video states [0, 2, 4].  Corresponding stored
    # action rows are [1, 3], which target states [2, 4].
    torch.testing.assert_close(sample["action"][0], torch.tensor([0.0] * 3 + [0.0, 0.0, 0.0, 1.0] + [0.0] * 65))
    torch.testing.assert_close(sample["action"][1], torch.tensor([2.0] * 3 + [0.0, 0.0, 0.0, 1.0] + [2.0] * 65))
    torch.testing.assert_close(sample["action"][2], torch.tensor([4.0] * 3 + [0.0, 0.0, 0.0, 1.0] + [4.0] * 65))
    assert sample["video"].shape == (3, 3, 8, 8)

    plan = build_sequence_plan_from_mode(
        mode="policy",
        video_length=sample["video"].shape[1],
        action_length=sample["action"].shape[0],
    )
    assert plan.condition_frame_indexes_vision == [0]
    assert plan.condition_frame_indexes_action == [0]
    assert plan.action_start_frame_offset == 0


def test_saber_meanstd_normalization_and_quaternion_postprocess(tmp_path, monkeypatch):
    root = _make_stream2_root(tmp_path)
    dataset = SABERG1LeRobotDataset(
        root=str(root),
        fps=30.0,
        chunk_length=2,
        split="full",
        action_normalization="meanstd",
    )
    monkeypatch.setattr(dataset, "_decode_video", _stub_video)
    sample = dataset[0]

    processor = ActionProcessor(max_action_dim=72)
    processed = processor.preprocess_action(
        {},
        sample["action"],
        action_normalizer=dataset.get_action_normalizer(sample),
    )
    torch.testing.assert_close(processed["action"][:, :3], sample["action"][:, :3] / 2.0)

    generated = processed["action"].clone()
    generated[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 0.25])
    recovered = processor.postprocess_action(generated, processed["action_processing_record"])
    torch.testing.assert_close(recovered[:, 3:7].norm(dim=-1), torch.ones(3))


def test_saber_domain_contract():
    assert get_domain_id("saber_g1") == 17
    assert get_action_dim("saber_g1") == 72


def test_saber_rejects_mismatched_state_action_names(tmp_path):
    root = _make_stream2_root(tmp_path)
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"]["action"]["names"][-1] = "wrong_channel"
    info_path.write_text(json.dumps(info))

    with pytest.raises(ValueError, match="channel names/order"):
        SABERG1LeRobotDataset(
            root=str(root),
            fps=30.0,
            chunk_length=2,
            split="full",
        )
