# Code taken from https://github.com/facebookresearch/segment-anything
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import namedtuple

from .bam import build_bam
from .hdmnet import build_hdmnet
from .la.build_lam import build_lam, build_lam_no_vit, build_lam_vit_mae_b, build_multilevel_lam, build_lam_vit_b_imagenet_i21k, build_lam_dino_b8, build_lam_vit_h, build_lam_vit_l, build_lam_vit_b
from .la.build_encoder import ENCODERS, build_vit_b, build_vit_h, build_vit_l
from .dcama import build_dcama
from .fptrans import build_fptrans
from .dmtnet import build_dmtnet


ComposedOutput = namedtuple("ComposedOutput", ["main", "aux"])

model_registry = {
    "lam": build_lam,
    "lam_no_vit": build_lam_no_vit,
    "lam_h": build_lam_vit_h,
    "lam_l": build_lam_vit_l,
    "lam_b": build_lam_vit_b,
    "lam_mae_b": build_lam_vit_mae_b,
    "lam_dino_b8": build_lam_dino_b8,
    "lam_b_imagenet_i21k": build_lam_vit_b_imagenet_i21k,
    "multilevel_lam": build_multilevel_lam,
    "dcama": build_dcama,
    "fptrans": build_fptrans,
    "bam": build_bam,
    "hdmnet": build_hdmnet,
    "dmtnet": build_dmtnet,
    # Encoders only
    **ENCODERS
}
