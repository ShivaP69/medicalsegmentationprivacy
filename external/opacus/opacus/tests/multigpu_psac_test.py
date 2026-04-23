#!/usr/bin/env python3
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

"""
Multi-GPU tests for PSACDPOptimizer (Per-Sample Adaptive Clipping) with distributed training.
"""

import os
import sys
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from opacus.grad_sample import GradSampleModule
from opacus.optimizers.ddppsacoptimizer import DistributedPSACDPOptimizer
from opacus.privacy_engine import PrivacyEngine
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler


def setup(rank, world_size):
    """Initialize distributed process group."""
    if sys.platform == "win32":
        raise ValueError("Windows platform is not supported for this test")
    else:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12356"

        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        torch.distributed.init_process_group(
            init_method="env://",
            backend="nccl",
        )


def cleanup():
    """Clean up distributed process group."""
    dist.destroy_process_group()


class ToyModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic(rank, weight, world_size, dp):
    """Demo function for distributed PSAC training."""
    torch.manual_seed(world_size)
    batch_size = 2
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = ToyModel().to(rank)
    model.net1.weight.data.zero_()
    optimizer = optim.SGD(model.parameters(), lr=1)

    # create dataset
    labels = torch.randn(2 * batch_size, 5).to(rank)
    data = torch.randn(2 * batch_size, 10)
    dataset = TensorDataset(data, labels)

    criterion = nn.CrossEntropyLoss(reduction="mean")

    max_grad_norm = 1.0

    # Wrap model with GradSampleModule
    model = GradSampleModule(model)
    ddp_model = DPDDP(model, device_ids=[rank])

    privacy_engine = PrivacyEngine()

    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    if dp:
        ddp_model, optimizer, data_loader = privacy_engine.make_private(
            module=ddp_model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=0,
            max_grad_norm=max_grad_norm,
            poisson_sampling=False,
            clipping="psac",
            r=0.01,
        )
        assert isinstance(optimizer, DistributedPSACDPOptimizer)

    for x, y in data_loader:
        outputs = ddp_model(x.to(rank))
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        break

    weight.copy_(model._module.net1.weight.data.cpu())
    cleanup()


def run_demo(demo_fn, weight, world_size, dp):
    """Run distributed demo."""
    mp.spawn(
        demo_fn,
        args=(weight, world_size, dp),
        nprocs=world_size,
        join=True,
    )


class GradientComputationTestPSAC(unittest.TestCase):
    """Test gradient computation correctness with PSAC in distributed setting."""

    @unittest.skipIf(torch.cuda.device_count() < 2, "Need at least 2 GPUs")
    def test_psac_gradient_correct(self) -> None:
        """Test that gradient is the same with DP or without DP in the distributed setting."""
        n_gpus = torch.cuda.device_count()
        self.assertTrue(
            n_gpus >= 2, f"Need at least 2 gpus but was provided only {n_gpus}."
        )

        weight_dp, weight_nodp = torch.ones(10, 10), torch.ones(10, 10)

        run_demo(
            demo_basic,
            weight_nodp,
            2,
            dp=False,
        )
        run_demo(
            demo_basic,
            weight_dp,
            2,
            dp=True,
        )

        self.assertTrue(torch.allclose(weight_dp, weight_nodp, atol=1e-5, rtol=1e-3))


if __name__ == "__main__":
    unittest.main()

