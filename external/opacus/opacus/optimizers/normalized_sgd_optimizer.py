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


class NormalizedSGDPOptimizer(DPOptimizer):
    """
    :class:`~opacus.optimizers.optimizer.DPOptimizer` that implements
    Normalized SGD with Perturbation for Differentially Private Non-Convex Optimization.

    This optimizer normalizes per-sample gradients instead of clipping them.
    For each per-sample gradient g^(i), it computes:
        g_normalized^(i) = g^(i) / (r + ||g^(i)||)
    where r > 0 is a regularization parameter.

    This approach is described in:
    "Differentially Private Non-Convex Optimization" (https://arxiv.org/pdf/2206.13033)

    The key difference from standard DP-SGD is that instead of hard clipping
    gradients to a threshold, this method normalizes them by their norm plus
    a regularization term, which can provide better convergence properties
    for non-convex optimization problems.

    Examples:
        >>> module = MyCustomModel()
        >>> optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        >>> dp_optimizer = NormalizedSGDPOptimizer(
        ...     optimizer=optimizer,
        ...     noise_multiplier=1.0,
        ...     regularization_param=0.1,
        ...     expected_batch_size=4,
        ... )
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        noise_multiplier: float,
        regularization_param: Optional[float] = None,
        max_grad_norm: Optional[float] = None,
        expected_batch_size: Optional[int],
        loss_reduction: str = "mean",
        generator=None,
        secure_mode: bool = False,
        **kwargs,
    ):
        """
        Args:
            optimizer: wrapped optimizer.
            noise_multiplier: noise multiplier for differential privacy.
                The noise standard deviation will be calibrated based on this
                and the expected gradient norm after normalization.
            regularization_param: regularization parameter r > 0 used in
                gradient normalization. Larger values provide more regularization
                and make the normalization less sensitive to gradient magnitude.
                If not provided, will use ``max_grad_norm`` as a fallback.
            max_grad_norm: if ``regularization_param`` is not provided, this will
                be used as the regularization parameter. This allows compatibility
                with PrivacyEngine which passes ``max_grad_norm``.
            expected_batch_size: batch_size used for averaging gradients. When using
                Poisson sampling averaging denominator can't be inferred from the
                actual batch size. Required if ``loss_reduction="mean"``, ignored if
                ``loss_reduction="sum"``
            loss_reduction: Indicates if the loss reduction (for aggregating the gradients)
                is a sum or a mean operation. Can take values "sum" or "mean"
            generator: torch.Generator() object used as a source of randomness for
                the noise
            secure_mode: if ``True`` uses noise generation approach robust to floating
                point arithmetic attacks.
                See :meth:`~opacus.optimizers.optimizer._generate_noise` for details
        """
        # Handle compatibility with PrivacyEngine which passes max_grad_norm
        if regularization_param is None:
            if max_grad_norm is None:
                raise ValueError(
                    "Either regularization_param or max_grad_norm must be provided"
                )
            regularization_param = max_grad_norm

        if regularization_param <= 0:
            raise ValueError(
                f"regularization_param must be positive, got {regularization_param}"
            )

        # For normalized SGD, we need to estimate the sensitivity.
        # After normalization, the gradient norm is bounded by 1 (since ||g||/(r+||g||) <= 1).
        # However, we use regularization_param as a proxy for the effective clipping bound
        # for noise calibration purposes. The actual sensitivity after normalization
        # is 1, but we scale noise by regularization_param to maintain similar
        # privacy-utility trade-offs as standard DP-SGD.
        effective_max_grad_norm = regularization_param

        super().__init__(
            optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=effective_max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
        )

        self.regularization_param = regularization_param

    def normalize_and_accumulate(self):
        """
        Performs gradient normalization (instead of clipping).
        Stores normalized and aggregated gradients into `p.summed_grad`.

        For each per-sample gradient g^(i), computes:
            g_normalized^(i) = g^(i) / (r + ||g^(i)||)
        where r is the regularization parameter.
        """
        if len(self.grad_samples[0]) == 0:
            # Empty batch
            per_sample_norm_factor = torch.zeros(
                (0,), device=self.grad_samples[0].device
            )
        else:
            per_param_norms = [
                g.reshape(len(g), -1).norm(2, dim=-1) for g in self.grad_samples
            ]

            if per_param_norms:
                target_device = per_param_norms[0].device
                per_param_norms = [norm.to(target_device) for norm in per_param_norms]

            per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
            # Normalization factor: 1 / (r + ||g||)
            # This normalizes the gradient instead of clipping it
            per_sample_norm_factor = 1.0 / (
                per_sample_norms + self.regularization_param
            )

        for p in self.params:
            _check_processed_flag(p.grad_sample)
            grad_sample = self._get_flat_grad_sample(p)

            # gradients should match the dtype of the optimizer parameters
            # for mixed precision, optimizer parameters are usually in FP32
            # lower precision grads will be cast up to FP32
            grad_sample = grad_sample.to(p.dtype)
            norm_factor_on_device = per_sample_norm_factor.to(grad_sample.device).to(
                p.dtype
            )
            # Apply normalization: g_normalized = g / (r + ||g||)
            grad = torch.einsum("i,i...", norm_factor_on_device, grad_sample)

            if p.summed_grad is not None:
                p.summed_grad += grad
            else:
                p.summed_grad = grad

            _mark_as_processed(p.grad_sample)

    def clip_and_accumulate(self):
        """
        Alias for normalize_and_accumulate to maintain compatibility
        with the DPOptimizer interface.
        """
        self.normalize_and_accumulate()

    def add_noise(self):
        """
        Adds noise to normalized gradients. Stores normalized and noised result in ``p.grad``.

        The noise standard deviation is calibrated as:
            std = noise_multiplier * regularization_param

        This ensures that the noise scale is appropriate for the normalized gradients.
        """
        for p in self.params:
            _check_processed_flag(p.summed_grad)

            # For normalized gradients, the sensitivity is effectively 1 (since
            # normalized gradient norm <= 1). We scale noise by regularization_param
            # to maintain similar privacy-utility trade-offs.
            noise = _generate_noise(
                std=self.noise_multiplier * self.regularization_param,
                reference=p.summed_grad,
                generator=self.generator,
                secure_mode=self.secure_mode,
            )
            p.grad = (p.summed_grad + noise).view_as(p)

            _mark_as_processed(p.summed_grad)
