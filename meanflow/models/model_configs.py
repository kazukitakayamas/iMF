# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

import copy

import torch.nn as nn
from models.meanflow import MeanFlow

from models.unet import SongUNet
from models.imf import iMF                      # [iMF] 追加
from models.imf_unet import iMFUNet             # [iMF] 追加

MODEL_ARCHS = {
    "unet": SongUNet,
    "imf_unet": iMFUNet,                        # [iMF] 追加
}

MODEL_CONFIGS = {
    "unet": {
        "img_resolution": 32,
        "in_channels": 3,
        "out_channels": 3,
        "channel_mult_noise": 2,
        "resample_filter": [1, 3, 3, 1],
        "channel_mult": [2, 2, 2],
        "encoder_type": "standard",
        "decoder_type": "standard",
    },
}
MODEL_CONFIGS["imf_unet"] = dict(MODEL_CONFIGS["unet"])   # [iMF] 追加 (同一バックボーン)

# [iMF] 追加: method -> (目的関数クラス, デフォルトの arch)
METHODS = {
    "mf": (MeanFlow, "unet"),
    "imf": (iMF, "imf_unet"),
}


def instantiate_model(args) -> nn.Module:
    method = getattr(args, "method", "mf")                       # [iMF] 追加
    assert method in METHODS, f"Unknown method {method}."         # [iMF] 追加
    model_cls, default_arch = METHODS[method]                     # [iMF] 追加

    architechture = getattr(args, "arch", None) or default_arch
    if method == "imf" and architechture == "unet":               # [iMF] 追加
        architechture = "imf_unet"                                # [iMF] 追加

    assert (
        architechture in MODEL_CONFIGS
    ), f"Model architecture {architechture} is missing its config."

    configs = copy.deepcopy(MODEL_CONFIGS[architechture])         # [iMF] 変更 (共有辞書の破壊を回避)
    configs['dropout'] = args.dropout
    arch = MODEL_ARCHS[architechture]
    if args.use_edm_aug:
        configs['augment_dim'] = 6

    if method == "imf":                                           # [iMF] 追加
        configs['num_classes'] = args.num_classes                 # [iMF] 追加
        configs['v_head'] = args.v_head                           # [iMF] 追加

    if hasattr(args, "model_channels") and args.model_channels:   # [iMF] 追加 (Colab 用の縮小)
        configs['model_channels'] = args.model_channels           # [iMF] 追加

    model = model_cls(arch=arch, net_configs=configs, args=args)  # [iMF] 変更

    return model
