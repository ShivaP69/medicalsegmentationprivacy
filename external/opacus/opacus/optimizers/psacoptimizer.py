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

import logging
from typing import Optional

import torch
from torch.optim import Optimizer

from .optimizer import (
    DPOptimizer,
    _check_processed_flag,
    _generate_noise,
    _mark_as_processed,
)

logger = logging.getLogger(__name__)


class PSACDPOptimizer(DPOptimizer):
    """
    :class:`~opacus.optimizers.optimizer.DPOptimizer` that implements
    Differentially Private Per-Sample Adaptive Clipping (DP-PSAC) algorithm
    based on the non-monotonic adaptive weight function from the paper
    "Differentially Private Learning with Per-Sample Adaptive Clipping".

    This optimizer uses per-sample adaptive clipping thresholds instead of
    a constant global clipping norm, which reduces the deviation between
    the clipped batch gradient and the true batch-averaged gradient.

    Unlike normalization-based approaches (Auto-S, NSGD) that use monotonic
    weight functions which over-weight small gradients, DP-PSAC uses a
    non-monotonic adaptive weight function that gives similar order of
    weights to samples with different gradient norms.

    Reference: "Differentially Private Learning with Per-Sample Adaptive Clipping"
    (https://arxiv.org/abs/2212.00328)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        noise_multiplier: float,
        max_grad_norm: float,
        expected_batch_size: Optional[int],
        r: float = 0.1,
        loss_reduction: str = "mean",
        generator=None,
        secure_mode: bool = False,
        **kwargs,
    ):
        """
        Args:
            optimizer: wrapped optimizer.
            noise_multiplier: noise multiplier for differential privacy.
            max_grad_norm: upper bound for per-sample clipping norms (C in the paper).
                This serves as a scaling factor for the adaptive clipping thresholds.
            expected_batch_size: batch_size used for averaging gradients. When using
                Poisson sampling, the averaging denominator can't be inferred from the
                actual batch size. Required if ``loss_reduction="mean"``, ignored if
                ``loss_reduction="sum"``.
            r: hyperparameter for the non-monotonic weight function (default: 0.01).
                Controls the adaptation level. The weight function is:
                w(||g||) = (||g|| + r) / (||g||^2 + r*||g|| + r)
                Typically r is set to 0.01 or 0.1. Smaller values give more adaptation.
            loss_reduction: Indicates if the loss reduction (for aggregating the gradients)
                is a sum or a mean operation. Can take values "sum" or "mean".
            generator: torch.Generator() object used as a source of randomness for
                the noise.
            secure_mode: if ``True`` uses noise generation approach robust to floating
                point arithmetic attacks.
                See :meth:`~opacus.optimizers.optimizer._generate_noise` for details.
        """
        super().__init__(
            optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
            **kwargs,
        )

        if r <= 0:
            raise ValueError("r must be positive")
        if max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")

        self.r = r
        self._per_sample_clip_norms = None

    def _compute_per_sample_adaptive_clip_norms(
        self, per_sample_norms: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute per-sample adaptive clipping thresholds using the non-monotonic
        weight function from the paper (Algorithm 1, line 5).

        The weight function is:
            w(||g||) = 1 / (||g|| + r/(||g|| + r))

        Which simplifies to:
            w(||g||) = (||g|| + r) / (||g||^2 + r*||g|| + r)

        The adaptive clipping threshold for each sample is:
            C_i = C * w(||g_i||)

        This ensures that samples with small gradients are weighted more (less
        clipped) and samples with large gradients are clipped more, while avoiding
        the over-weighting problem of monotonic functions like 1/(||g|| + r).

        Args:
            per_sample_norms: per-sample gradient norms, shape [batch_size]

        Returns:
            Adaptive clipping thresholds for each sample, shape [batch_size]
        """
        r = self.r
        norms = per_sample_norms

        # Compute the non-monotonic weight function
        # w(||g||) = (||g|| + r) / (||g||^2 + r*||g|| + r)
        numerator = norms + r
        denominator = norms * norms + r * norms + r
        weight = numerator / denominator

        # Adaptive clipping threshold: C_i = C * w(||g_i||)
        adaptive_clip_norms = self.max_grad_norm * weight

        # Clamp to reasonable bounds for numerical stability
        min_clip_norm = self.max_grad_norm * r / (1.0 + r)
        adaptive_clip_norms = torch.clamp(
            adaptive_clip_norms, min=min_clip_norm, max=self.max_grad_norm
        )

        return adaptive_clip_norms

    def clip_and_accumulate(self):
        """
        Performs per-sample adaptive gradient clipping and accumulation.

        Steps:
        1. Compute per-sample gradient norms across all parameters
        2. Compute adaptive clipping thresholds using the paper's weight function
        3. Compute per-sample clip factors: min(1, C_i / ||g_i||)
        4. Apply clip factors and accumulate clipped gradients
        """
        if len(self.grad_samples) == 0 or len(self.grad_samples[0]) == 0:
            # Empty batch case
            per_sample_clip_factor = torch.zeros(
                (0,),
                device=(
                    self.grad_samples[0].device
                    if self.grad_samples
                    else torch.device("cpu")
                ),
            )
            self._per_sample_clip_norms = None
        else:
            # Step 1: Compute per-parameter norms for each sample
            per_param_norms = [
                g.reshape(len(g), -1).norm(2, dim=-1) for g in self.grad_samples
            ]

            if per_param_norms:
                target_device = per_param_norms[0].device
                per_param_norms = [norm.to(target_device) for norm in per_param_norms]

            # Step 2: Compute per-sample gradient norms (L2 norm across all parameters)
            per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)

            # Step 3: Compute per-sample adaptive clipping thresholds
            self._per_sample_clip_norms = self._compute_per_sample_adaptive_clip_norms(
                per_sample_norms
            )

            # Step 4: Compute per-sample clip factors: min(1, C_i / ||g_i||)
            # The .clamp(max=1.0) ensures we only clip down, never scale up
            per_sample_clip_factor = (
                self._per_sample_clip_norms / (per_sample_norms + 1e-8)
            ).clamp(max=1.0)

        # Apply clipping to each parameter's gradients and accumulate
        for p in self.params:
            _check_processed_flag(p.grad_sample)
            grad_sample = self._get_flat_grad_sample(p)

            # Convert to parameter dtype
            grad_sample = grad_sample.to(p.dtype)
            clip_factor_on_device = per_sample_clip_factor.to(grad_sample.device).to(
                p.dtype
            )

            # Apply per-sample clip factors and sum across samples
            # einsum operation: for each sample i, scale its gradients by clip_factor[i]
            grad = torch.einsum("i,i...", clip_factor_on_device, grad_sample)

            if p.summed_grad is not None:
                p.summed_grad += grad
            else:
                p.summed_grad = grad

            _mark_as_processed(p.grad_sample)

    def add_noise(self):
        """
        Adds Gaussian noise to clipped gradients to achieve differential privacy.

        The noise is calibrated based on the maximum clipping norm to bound the
        sensitivity of the gradient computation. Even though individual samples have
        different adaptive thresholds C_i, the global sensitivity is still bounded
        by C = max_grad_norm because each clipped gradient satisfies ||clipped_g_i|| ≤ C.

        This maintains the (ε, δ)-differential privacy guarantee.
        """
        # Determine the effective clipping norm for noise calibration
        if (
            self._per_sample_clip_norms is not None
            and len(self._per_sample_clip_norms) > 0
        ):
            max_adaptive_clip_norm = self._per_sample_clip_norms.max().item()
            effective_clip_norm = min(max_adaptive_clip_norm, self.max_grad_norm)
        else:
            effective_clip_norm = self.max_grad_norm

        # Add calibrated noise to each parameter's summed gradient
        for p in self.params:
            _check_processed_flag(p.summed_grad)

            noise = _generate_noise(
                std=self.noise_multiplier * effective_clip_norm,
                reference=p.summed_grad,
                generator=self.generator,
                secure_mode=self.secure_mode,
            )
            p.grad = (p.summed_grad + noise).view_as(p)

            _mark_as_processed(p.summed_grad)

    def zero_grad(self, set_to_none: bool = False):
        """
        Clear gradients and per-sample adaptive clipping state.
        """
        super().zero_grad(set_to_none)
        self._per_sample_clip_norms = None
