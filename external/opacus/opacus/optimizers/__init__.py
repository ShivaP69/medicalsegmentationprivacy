# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .adaclipoptimizer import AdaClipDPOptimizer
from .ddp_normalized_sgd_optimizer import DistributedNormalizedSGDPOptimizer
from .ddp_perlayeroptimizer import SimpleDistributedPerLayerOptimizer
from .ddpoptimizer import DistributedDPOptimizer
from .ddpoptimizer_automatic_clipping import (
    DistributedDPAutomaticClippingOptimizer,
    DistributedDPPerLayerAutomaticClippingOptimizer,
)
from .ddpoptimizer_fast_gradient_clipping import (
    DistributedDPOptimizerFastGradientClipping,
)
from .ddppsacoptimizer import DistributedPSACDPOptimizer
from .fsdpoptimizer_fast_gradient_clipping import FSDPOptimizerFastGradientClipping
from .normalized_sgd_optimizer import NormalizedSGDPOptimizer
from .optimizer import DPOptimizer
from .optimizer_automatic_clipping import (
    DPAutomaticClippingOptimizer,
    DPPerLayerAutomaticClippingOptimizer,
)
from .optimizer_fast_gradient_clipping import DPOptimizerFastGradientClipping
from .perlayeroptimizer import DPPerLayerOptimizer
from .psacoptimizer import PSACDPOptimizer


__all__ = [
    "AdaClipDPOptimizer",
    "DistributedDPOptimizer",
    "DistributedPSACDPOptimizer",
    "DistributedNormalizedSGDPOptimizer",
    "DPOptimizer",
    "DPOptimizerFastGradientClipping",
    "DistributedDPOptimizerFastGradientlipping",
    "FSDPOptimizerFastGradientClipping",
    "NormalizedSGDPOptimizer",
    "DPPerLayerOptimizer",
    "PSACDPOptimizer",
    "SimpleDistributedPerLayerOptimizer",
    "DPAutomaticClippingOptimizer",
    "DPPerLayerAutomaticClippingOptimizer",
    "DistributedDPAutomaticClippingOptimizer",
    "DistributedDPPerLayerAutomaticClippingOptimizer",
]


def get_optimizer_class(clipping: str, distributed: bool, grad_sample_mode: str = None):
    if grad_sample_mode == "ghost":
        if clipping == "flat" and distributed is False:
            return DPOptimizerFastGradientClipping
        elif clipping == "flat" and distributed is True:
            return DistributedDPOptimizerFastGradientClipping
        else:
            raise ValueError(
                f"Unsupported combination of parameters. Clipping: {clipping} and grad_sample_mode: {grad_sample_mode}"
            )
    elif grad_sample_mode == "ghost_fsdp":
        if clipping == "flat" and distributed is True:
            return FSDPOptimizerFastGradientClipping
        else:
            raise ValueError(
                f"Unsupported combination of parameters. Clipping: {clipping}, distributed: {distributed}, and grad_sample_mode: {grad_sample_mode}"
            )
    elif clipping == "flat" and distributed is False:
        return DPOptimizer
    elif clipping == "flat" and distributed is True:
        return DistributedDPOptimizer
    elif clipping == "per_layer" and distributed is False:
        return DPPerLayerOptimizer
    elif clipping == "per_layer" and distributed is True:
        if grad_sample_mode == "hooks" or grad_sample_mode == "ew":
            return SimpleDistributedPerLayerOptimizer
        else:
            raise ValueError(f"Unexpected grad_sample_mode: {grad_sample_mode}")
    elif clipping == "automatic" and distributed is False:
        return DPAutomaticClippingOptimizer
    elif clipping == "automatic" and distributed is True:
        return DistributedDPAutomaticClippingOptimizer
    elif clipping == "automatic_per_layer" and distributed is False:
        return DPPerLayerAutomaticClippingOptimizer
    elif clipping == "automatic_per_layer" and distributed is True:
        return DistributedDPPerLayerAutomaticClippingOptimizer
    elif clipping == "adaptive" and distributed is False:
        return AdaClipDPOptimizer
    elif clipping == "psac" and distributed is False:
        return PSACDPOptimizer
    elif clipping == "psac" and distributed is True:
        return DistributedPSACDPOptimizer
    elif clipping == "normalized_sgd" and distributed is False:
        return NormalizedSGDPOptimizer
    elif clipping == "normalized_sgd" and distributed is True:
        return DistributedNormalizedSGDPOptimizer
    raise ValueError(
        f"Unexpected optimizer parameters. Clipping: {clipping}, distributed: {distributed}"
    )
