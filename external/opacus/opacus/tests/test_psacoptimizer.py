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
Tests for PSACDPOptimizer (Per-Sample Adaptive Clipping optimizer).
"""

import unittest

import torch
import torch.nn as nn
from opacus.grad_sample import GradSampleModule
from opacus.optimizers.psacoptimizer import PSACDPOptimizer


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class PSACOptimizerTest(unittest.TestCase):
    """Test suite for PSACDPOptimizer."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = SimpleModel()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.batch_size = 4

    def test_init(self):
        """Test PSACDPOptimizer initialization."""
        dp_optimizer = PSACDPOptimizer(
            optimizer=self.optimizer,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            expected_batch_size=self.batch_size,
            r=0.01,
        )

        self.assertEqual(dp_optimizer.r, 0.01)
        self.assertEqual(dp_optimizer.max_grad_norm, 1.0)
        self.assertEqual(dp_optimizer.noise_multiplier, 1.0)

    def test_init_invalid_r(self):
        """Test that invalid r raises ValueError."""
        with self.assertRaises(ValueError):
            PSACDPOptimizer(
                optimizer=self.optimizer,
                noise_multiplier=1.0,
                max_grad_norm=1.0,
                expected_batch_size=self.batch_size,
                r=-0.01,
            )


    def test_compute_per_sample_adaptive_clip_norms(self):
        """Test per-sample adaptive clipping norm computation."""
        dp_optimizer = PSACDPOptimizer(
            optimizer=self.optimizer,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            expected_batch_size=self.batch_size,
            r=0.01,
        )

        per_sample_norms = torch.tensor([0.05, 0.3, 0.6, 1.0])
        clip_norms = dp_optimizer._compute_per_sample_adaptive_clip_norms(
            per_sample_norms
        )

        # Clip norms should be within bounds
        self.assertTrue(torch.all(clip_norms > 0))
        self.assertTrue(torch.all(clip_norms <= dp_optimizer.max_grad_norm))

        # Each sample should have its own adaptive threshold
        self.assertEqual(len(clip_norms), len(per_sample_norms))
        
        # Test that the weight function is non-monotonic
        # For small norms, weight should be higher (less clipping)
        # For large norms, weight should be lower (more clipping)
        small_norm_clip = clip_norms[0] / dp_optimizer.max_grad_norm
        large_norm_clip = clip_norms[-1] / dp_optimizer.max_grad_norm
        # Small norm should have higher relative clip threshold
        self.assertGreater(small_norm_clip, large_norm_clip)

    def test_clip_and_accumulate(self):
        """Test clip_and_accumulate with per-sample adaptive clipping."""
        model = GradSampleModule(SimpleModel())
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        dp_optimizer = PSACDPOptimizer(
            optimizer=optimizer,
            noise_multiplier=0.0,  # No noise for testing
            max_grad_norm=1.0,
            expected_batch_size=self.batch_size,
            r=0.01,
        )

        # Create fake per-sample gradients
        for p in model.parameters():
            if p.requires_grad:
                p.grad_sample = torch.randn(
                    self.batch_size, *p.shape, device=p.device
                )

        # Should not raise any errors
        dp_optimizer.clip_and_accumulate()

        # Check that summed_grad was created
        for p in model.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.summed_grad)
                self.assertEqual(p.summed_grad.shape, p.shape)

        # Check that per-sample clip norms were computed
        self.assertIsNotNone(dp_optimizer._per_sample_clip_norms)

    def test_clip_and_accumulate_empty_batch(self):
        """Test clip_and_accumulate with empty batch."""
        model = GradSampleModule(SimpleModel())
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        dp_optimizer = PSACDPOptimizer(
            optimizer=optimizer,
            noise_multiplier=0.0,
            max_grad_norm=1.0,
            expected_batch_size=0,
            r=0.01,
        )

        # Create empty per-sample gradients
        for p in model.parameters():
            if p.requires_grad:
                p.grad_sample = torch.empty(0, *p.shape, device=p.device)

        # Should not raise any errors
        dp_optimizer.clip_and_accumulate()

    def test_add_noise(self):
        """Test noise addition."""
        model = GradSampleModule(SimpleModel())
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        dp_optimizer = PSACDPOptimizer(
            optimizer=optimizer,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            expected_batch_size=self.batch_size,
            r=0.01,
        )

        # Create fake per-sample gradients and clip
        for p in model.parameters():
            if p.requires_grad:
                p.grad_sample = torch.randn(
                    self.batch_size, *p.shape, device=p.device
                )

        dp_optimizer.clip_and_accumulate()
        dp_optimizer.add_noise()

        # Check that grad was created with noise
        for p in model.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad)
                self.assertEqual(p.grad.shape, p.shape)

    def test_zero_grad(self):
        """Test zero_grad clears per-sample clipping state."""
        dp_optimizer = PSACDPOptimizer(
            optimizer=self.optimizer,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            expected_batch_size=self.batch_size,
            r=0.01,
        )

        # Set some state
        dp_optimizer._per_sample_clip_norms = torch.tensor([1.0, 2.0])

        dp_optimizer.zero_grad()

        # Should clear per-sample clip norms
        self.assertIsNone(dp_optimizer._per_sample_clip_norms)

    def test_full_step(self):
        """Test full optimizer step."""
        model = GradSampleModule(SimpleModel())
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        dp_optimizer = PSACDPOptimizer(
            optimizer=optimizer,
            noise_multiplier=0.0,  # No noise for deterministic test
            max_grad_norm=1.0,
            expected_batch_size=self.batch_size,
            r=0.01,
        )

        # Store initial parameter values
        initial_params = {name: p.data.clone() for name, p in model.named_parameters()}

        # Create fake per-sample gradients
        for p in model.parameters():
            if p.requires_grad:
                p.grad_sample = torch.randn(
                    self.batch_size, *p.shape, device=p.device
                )

        # Full step should work
        dp_optimizer.step()

        # Parameters should have been updated (changed from initial values)
        for name, p in model.named_parameters():
            if p.requires_grad:
                # Parameters should have changed after step
                self.assertFalse(
                    torch.equal(p.data, initial_params[name]),
                    f"Parameter {name} was not updated after step"
                )
                # grad_sample is only cleared by zero_grad(), not automatically
                self.assertIsNotNone(p.grad_sample)

    def test_per_sample_adaptive_clipping_different_norms(self):
        """Test that different gradient norms get different clipping thresholds."""
        dp_optimizer = PSACDPOptimizer(
            optimizer=self.optimizer,
            noise_multiplier=0.0,
            max_grad_norm=1.0,
            expected_batch_size=self.batch_size,
            r=0.01,
        )

        # Create gradients with very different norms
        per_sample_norms = torch.tensor([0.01, 0.2, 0.8, 2.0])
        clip_norms = dp_optimizer._compute_per_sample_adaptive_clip_norms(
            per_sample_norms
        )

        # Different norms should result in different clip thresholds
        # (though they may be similar due to the weight function)
        self.assertTrue(len(set(clip_norms.tolist())) > 1)
        
        # Verify the weight function behavior: smaller norms get higher clip thresholds
        # (relative to their norm size)
        for i in range(len(per_sample_norms) - 1):
            norm_ratio = per_sample_norms[i] / (per_sample_norms[i+1] + 1e-8)
            clip_ratio = clip_norms[i] / (clip_norms[i+1] + 1e-8)
            # Smaller norm should have relatively higher clip threshold
            if norm_ratio < 1.0:
                self.assertGreater(clip_ratio, norm_ratio)

    def test_default_parameters(self):
        """Test that default parameters work correctly."""
        dp_optimizer = PSACDPOptimizer(
            optimizer=self.optimizer,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            expected_batch_size=self.batch_size,
        )

        # Default values should be set
        self.assertEqual(dp_optimizer.r, 0.01)


if __name__ == "__main__":
    unittest.main()

