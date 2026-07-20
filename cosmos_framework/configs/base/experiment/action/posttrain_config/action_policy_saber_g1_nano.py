# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Cosmos3-Nano action-policy SFT recipe for DreamVu SABER Stream 2."""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_droid_nano import (
    action_policy_droid_nano,
)
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import get_action_saber_g1_sft_dataset
from cosmos_framework.utils.lazy_config import LazyCall as L

cs = ConfigStore.instance()


# Reuse the validated DROID action-policy trainer/checkpoint stack, then replace
# only the model width and data contract that differ for SABER's Unitree G1.
action_policy_saber_g1_nano = copy.deepcopy(action_policy_droid_nano)
action_policy_saber_g1_nano["job"].update(
    group="action_sft",
    name="action_policy_saber_g1_nano",
)

model_config = action_policy_saber_g1_nano["model"]["config"]
model_config["max_action_dim"] = 72
model_config["resolution"] = "480"
model_config["tokenizer"]["encode_exact_durations"] = [33]
model_config["max_num_tokens_after_packing"] = -1
model_config["rectified_flow_training_config"]["loss_scale"] = 10.0

# Stream 2 is much smaller than DROID.  Start from the LIBERO-scale shared
# learning rate while retaining the validated 5x multiplier for fresh action
# projections.
action_policy_saber_g1_nano["optimizer"]["lr"] = 5.0e-05
action_policy_saber_g1_nano["scheduler"].update(
    cycle_lengths=[5000],
    f_max=[1.0],
    f_min=[0.0],
    f_start=[1.0e-06],
    warm_up_steps=[500],
)

action_policy_saber_g1_nano["dataloader_train"]["dataset_name"] = "action_saber_g1"
action_policy_saber_g1_nano["dataloader_train"]["max_samples_per_batch"] = 32
action_policy_saber_g1_nano["dataloader_train"]["dataloader"]["datasets"] = {
    "saber_g1": {
        "ratio": 1,
        "dataset": L(get_action_saber_g1_sft_dataset)(
            root="${oc.env:SABER_ROOT}",
            fps=15.0,
            chunk_length=32,
            mode="policy",
            split="train",
            val_ratio=0.05,
            seed=42,
            action_normalization="meanstd",
            sample_stride=1,
            resolution="480",
            max_action_dim="${model.config.max_action_dim}",
            tokenizer_config="${model.config.vlm_config.tokenizer}",
            cfg_dropout_rate=0.1,
            format_prompt_as_json=True,
            iterable_shuffle=True,
            episode_shuffle_seed=42,
        ),
    }
}


cs.store(
    group="experiment",
    package="_global_",
    name="action_policy_saber_g1_nano",
    node=action_policy_saber_g1_nano,
)
