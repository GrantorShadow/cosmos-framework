# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Count parameters in a DCP checkpoint by Cosmos optimizer-selection substrings."""

from __future__ import annotations

import argparse
import json
import math

from torch.distributed.checkpoint import FileSystemReader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_model_path")
    args = parser.parse_args()

    metadata = FileSystemReader(args.checkpoint_model_path).read_metadata()
    selected_substrings = [
        "moe_gen",
        "time_embedder",
        "vae2llm",
        "llm2vae",
        "action2llm",
        "llm2action",
        "action_modality_embed",
    ]
    totals = {key: 0 for key in selected_substrings}
    all_parameters = 0
    selected_parameters = 0
    selected_tensors = 0
    for name, tensor_metadata in metadata.state_dict_metadata.items():
        size = getattr(tensor_metadata, "size", None)
        if size is None:
            continue
        count = math.prod(size)
        all_parameters += count
        matches = [key for key in selected_substrings if key in name]
        if matches:
            selected_parameters += count
            selected_tensors += 1
            for key in matches:
                totals[key] += count

    print(
        json.dumps(
            {
                "all_parameters": all_parameters,
                "selected_parameters": selected_parameters,
                "selected_tensors": selected_tensors,
                "selected_by_substring": totals,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
