#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Cosmos3-Nano policy SFT on DreamVu SABER-10K/SABER-stream2.
#
# DATASET_PATH may point at SABER-stream2 itself or the SABER-10K parent.
# BASE_CHECKPOINT_PATH must be a converted Cosmos3-Nano DCP checkpoint.

: "${TOML_FILE:=examples/toml/sft_config/action_policy_saber_g1.toml}"
: "${DATASET_PATH:=${SABER_ROOT:-examples/data/SABER-10K/SABER-stream2}}"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Nano}"

EXTRA_DATASET_CHECK='
export SABER_ROOT="$DATASET_PATH"
_saber_root="$SABER_ROOT"
if [[ ! -f "$_saber_root/meta/info.json" && -f "$_saber_root/SABER-stream2/meta/info.json" ]]; then
    _saber_root="$_saber_root/SABER-stream2"
fi
[[ -f "$_saber_root/meta/info.json" ]] ||
    { echo "ERROR: SABER_ROOT must contain SABER-stream2/meta/info.json (got: $SABER_ROOT)" >&2; exit 1; }
[[ -f "$_saber_root/meta/modality.json" && -f "$_saber_root/meta/stats.json" ]] ||
    { echo "ERROR: SABER Stream 2 metadata is incomplete under $_saber_root/meta" >&2; exit 1; }
'

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
