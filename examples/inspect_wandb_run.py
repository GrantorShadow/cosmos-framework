# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Summarize loss, gradient, and learning-rate histories for a Cosmos W&B run."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics

import wandb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--project", default="cosmos3_saber")
    args = parser.parse_args()

    entity = os.environ.get("WANDB_ENTITY", "tristar-ai")
    run = wandb.Api(timeout=60).run(f"{entity}/{args.project}/{args.run_id}")
    rows = list(run.scan_history())
    series: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            key_lower = key.lower()
            if not any(token in key_lower for token in ("loss", "grad_norm", "optim/lr")):
                continue
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                series.setdefault(key, []).append(float(value))

    summaries: dict[str, dict[str, float | int]] = {}
    for key, values in sorted(series.items()):
        if len(values) < 2:
            continue
        quarter = max(1, len(values) // 4)
        summaries[key] = {
            "count": len(values),
            "first": values[0],
            "last": values[-1],
            "min": min(values),
            "max": max(values),
            "mean": statistics.fmean(values),
            "first_quarter_mean": statistics.fmean(values[:quarter]),
            "last_quarter_mean": statistics.fmean(values[-quarter:]),
        }

    print(f"W&B run: {run.url}")
    print(f"History rows: {len(rows)}")
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
