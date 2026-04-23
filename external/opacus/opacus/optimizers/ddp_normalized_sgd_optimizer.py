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

from .normalized_sgd_optimizer import NormalizedSGDPOptimizer
from .optimizer import _check_processed_flag, _mark_as_processed


class DistributedNormalizedSGDPOptimizer(NormalizedSGDPOptimizer):
    """
    :class:`~opacus.optimizers.normalized_sgd_optimizer.NormalizedSGDPOptimizer`
    compatible with distributed data processing.

    This optimizer extends NormalizedSGDPOptimizer to work in distributed training
    settings where gradients need to be synchronized across multiple workers.
    Noise is only added on the first worker (rank 0) to maintain privacy guarantees
    while avoiding redundant noise addition.

    Examples:
        >>> module = MyCustomModel()
        >>> optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        >>> dp_optimizer = DistributedNormalizedSGDPOptimizer(
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
        super().__init__(
            optimizer,
            noise_multiplier=noise_multiplier,
            regularization_param=regularization_param,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
            **kwargs,
        )
        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()

    def add_noise(self):
        """
        Adds noise to normalized gradients. Only adds noise on rank 0 worker
        to maintain privacy guarantees while avoiding redundant noise addition.

        After noise is added on rank 0, gradients will be synchronized
        across all workers via reduce_gradients().
        """
        # Noise only gets added to the first worker
        if self.rank == 0:
            super().add_noise()
        else:
            for p in self.params:
                _check_processed_flag(p.summed_grad)
                p.grad = p.summed_grad.view_as(p)
                _mark_as_processed(p.summed_grad)

    def reduce_gradients(self):
        """
        Synchronizes gradients across all workers using all_reduce.

        After normalization and noise addition (on rank 0), gradients are
        summed across all workers and optionally averaged if loss_reduction="mean".
        """
        for p in self.params:
            if not p.requires_grad:
                continue
            torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.SUM)
            if self.loss_reduction == "mean":
                p.grad /= self.world_size

    def step(
        self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[torch.Tensor]:
        """
        Performs a single optimization step with distributed gradient synchronization.

        The step process:
        1. Normalize and accumulate per-sample gradients (on each worker)
        2. Add noise (only on rank 0)
        3. Scale gradients by loss_reduction
        4. Synchronize gradients across all workers
        5. Call underlying optimizer step

        Args:
            closure: A closure that reevaluates the model and returns the loss.
                Optional for most optimizers.

        Returns:
            The loss value if closure is provided, None otherwise.
        """
        if closure is not None:
            with torch.enable_grad():
                closure()

        if self.pre_step():
            self.reduce_gradients()
            return self.original_optimizer.step()
        else:
            return None
