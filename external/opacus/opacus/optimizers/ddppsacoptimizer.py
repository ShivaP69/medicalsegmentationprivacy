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

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch.optim import Optimizer

from .ddpoptimizer import DistributedDPOptimizer
from .psacoptimizer import PSACDPOptimizer


class DistributedPSACDPOptimizer(PSACDPOptimizer, DistributedDPOptimizer):
    """
    :class:`~opacus.optimizers.psacoptimizer.PSACDPOptimizer` compatible with
    distributed data parallel training.

    This optimizer combines per-sample adaptive clipping (DP-PSAC) with
    distributed data parallel processing, allowing for efficient multi-GPU
    training with differential privacy.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        noise_multiplier: float,
        max_grad_norm: float,
        expected_batch_size: Optional[int],
        r: float = 0.01,
        loss_reduction: str = "mean",
        generator=None,
        secure_mode: bool = False,
        **kwargs,
    ):
        """
        Args:
            optimizer: wrapped optimizer.
            noise_multiplier: noise multiplier for differential privacy
            max_grad_norm: upper bound for per-sample clipping norms (C in the paper).
                This serves as a scaling factor for the adaptive clipping thresholds.
            expected_batch_size: batch_size used for averaging gradients. When using
                Poisson sampling averaging denominator can't be inferred from the
                actual batch size. Required if ``loss_reduction="mean"``, ignored if
                ``loss_reduction="sum"``
            r: hyperparameter for the non-monotonic weight function (default: 0.01).
                Controls the adaptation level. The weight function is:
                w(||g||) = (||g|| + r) / (||g||^2 + r*||g|| + r)
                Typically r is set to 0.01 or 0.1. Smaller values give more adaptation.
            loss_reduction: Indicates if the loss reduction (for aggregating the gradients)
                is a sum or a mean operation. Can take values "sum" or "mean"
            generator: torch.Generator() object used as a source of randomness for
                the noise
            secure_mode: if ``True`` uses noise generation approach robust to floating
                point arithmetic attacks.
                See :meth:`~opacus.optimizers.optimizer._generate_noise` for details
        """
        # Call PSACDPOptimizer.__init__ which will call DPOptimizer.__init__
        # Then set up distributed attributes
        PSACDPOptimizer.__init__(
            self,
            optimizer=optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            r=r,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
            **kwargs,
        )
        # Set up distributed attributes (rank and world_size)
        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()

    def add_noise(self):
        """
        Adds noise to clipped gradients. In distributed setting, noise is only
        added on rank 0 to avoid redundant noise addition.
        """
        # Noise only gets added to the first worker
        if self.rank == 0:
            PSACDPOptimizer.add_noise(self)
        else:
            for p in self.params:
                p.grad = p.summed_grad.view_as(p)

    def step(
        self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[torch.Tensor]:
        """
        Performs a single optimization step with distributed gradient reduction.
        """
        if closure is not None:
            with torch.enable_grad():
                closure()

        if self.pre_step():
            self.reduce_gradients()
            return self.original_optimizer.step()
        else:
            return None
