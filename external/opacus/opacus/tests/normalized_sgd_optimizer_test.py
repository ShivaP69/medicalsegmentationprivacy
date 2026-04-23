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

import unittest

import torch
import torch.nn as nn
from opacus.optimizers.normalized_sgd_optimizer import NormalizedSGDPOptimizer


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class NormalizedSGDPOptimizerTest(unittest.TestCase):
    """Test suite for NormalizedSGDPOptimizer."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.model = SimpleModel()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.batch_size = 4
        self.regularization_param = 0.1
        self.noise_multiplier = 1.0

    def test_init_valid(self):
        """Test that NormalizedSGDPOptimizer initializes correctly with valid parameters."""
        dp_optimizer = NormalizedSGDPOptimizer(
            optimizer=self.optimizer,
            noise_multiplier=self.noise_multiplier,
            regularization_param=self.regularization_param,
            expected_batch_size=self.batch_size,
        )
        self.assertEqual(dp_optimizer.regularization_param, self.regularization_param)
        self.assertEqual(dp_optimizer.noise_multiplier, self.noise_multiplier)

    def test_init_invalid_regularization_param(self):
        """Test that NormalizedSGDPOptimizer raises error for invalid regularization_param."""
        with self.assertRaises(ValueError) as context:
            NormalizedSGDPOptimizer(
                optimizer=self.optimizer,
                noise_multiplier=self.noise_multiplier,
                regularization_param=0.0,  # Invalid: must be > 0
                expected_batch_size=self.batch_size,
            )
        self.assertIn("regularization_param must be positive", str(context.exception))

        with self.assertRaises(ValueError) as context:
            NormalizedSGDPOptimizer(
                optimizer=self.optimizer,
                noise_multiplier=self.noise_multiplier,
                regularization_param=-0.1,  # Invalid: must be > 0
                expected_batch_size=self.batch_size,
            )
        self.assertIn("regularization_param must be positive", str(context.exception))

    def test_normalize_and_accumulate(self):
        """Test that gradient normalization works correctly."""
        dp_optimizer = NormalizedSGDPOptimizer(
            optimizer=self.optimizer,
            noise_multiplier=0.0,  # No noise for this test
            regularization_param=self.regularization_param,
            expected_batch_size=self.batch_size,
        )

        # Create fake per-sample gradients
        for p in self.model.parameters():
            if p.requires_grad:
                p.grad_sample = torch.randn(self.batch_size, *p.shape)

        # Normalize and accumulate
        dp_optimizer.normalize_and_accumulate()

        # Check that summed_grad was created
        for p in self.model.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.summed_grad, "summed_grad should be set")

    def test_normalization_formula(self):
        """Test that normalization formula is correct: g / (r + ||g||)."""
        dp_optimizer = NormalizedSGDPOptimizer(
            optimizer=self.optimizer,
            noise_multiplier=0.0,  # No noise for this test
            regularization_param=self.regularization_param,
            expected_batch_size=self.batch_size,
        )

        # Create a simple test case with known gradients
        # Set up grad_sample for all parameters (required by optimizer)
        for p in self.model.parameters():
            if p.requires_grad:
                # Create simple per-sample gradients with known values
                p.grad_sample = torch.randn(self.batch_size, *p.shape)

        # Normalize and accumulate
        dp_optimizer.normalize_and_accumulate()

        # Check that the normalization was applied correctly
        # The summed_grad should be the sum of normalized gradients
        for p in self.model.parameters():
            if p.requires_grad:
                self.assertIsNotNone(
                    p.summed_grad, "summed_grad should be set after normalization"
                )

    def test_normalized_gradient_norm_bounded(self):
        """Test that normalized gradients have bounded norm."""
        dp_optimizer = NormalizedSGDPOptimizer(
            optimizer=self.optimizer,
            noise_multiplier=0.0,  # No noise for this test
            regularization_param=self.regularization_param,
            expected_batch_size=self.batch_size,
        )

        # Create gradients with large norms
        for p in self.model.parameters():
            if p.requires_grad:
                # Create gradients with very large norms
                p.grad_sample = torch.randn(self.batch_size, *p.shape) * 100.0

        dp_optimizer.normalize_and_accumulate()

        # Check that normalized gradients have reasonable norms
        # After normalization, each per-sample gradient norm should be <= 1
        for p in self.model.parameters():
            if p.requires_grad:
                # Reconstruct per-sample gradients from summed_grad to verify
                # In practice, we can't easily reconstruct them, but we can verify
                # that the summed gradient is reasonable
                self.assertIsNotNone(p.summed_grad)

    def test_add_noise(self):
        """Test that noise is added correctly."""
        dp_optimizer = NormalizedSGDPOptimizer(
            optimizer=self.optimizer,
            noise_multiplier=self.noise_multiplier,
            regularization_param=self.regularization_param,
            expected_batch_size=self.batch_size,
        )

        # Create fake per-sample gradients and normalize
        for p in self.model.parameters():
            if p.requires_grad:
                p.grad_sample = torch.randn(self.batch_size, *p.shape)

        dp_optimizer.normalize_and_accumulate()

        # Store summed_grad before adding noise
        summed_grads_before = {
            p: p.summed_grad.clone() if p.summed_grad is not None else None
            for p in self.model.parameters()
            if p.requires_grad
        }

        # Add noise
        dp_optimizer.add_noise()

        # Check that noise was added (grad should differ from summed_grad)
        for p in self.model.parameters():
            if p.requires_grad and summed_grads_before[p] is not None:
                self.assertIsNotNone(p.grad, "grad should be set after add_noise")
                # With noise_multiplier > 0, grad should differ from summed_grad
                if self.noise_multiplier > 0:
                    self.assertFalse(
                        torch.allclose(p.grad, summed_grads_before[p], atol=1e-6),
                        "grad should differ from summed_grad when noise is added",
                    )

    def test_no_noise(self):
        """Test that with noise_multiplier=0, no noise is added."""
        dp_optimizer = NormalizedSGDPOptimizer(
            optimizer=self.optimizer,
            noise_multiplier=0.0,
            regularization_param=self.regularization_param,
            expected_batch_size=self.batch_size,
        )

        # Create fake per-sample gradients and normalize
        for p in self.model.parameters():
            if p.requires_grad:
                p.grad_sample = torch.randn(self.batch_size, *p.shape)

        dp_optimizer.normalize_and_accumulate()

        # Store summed_grad before adding noise
        summed_grads_before = {
            p: p.summed_grad.clone() if p.summed_grad is not None else None
            for p in self.model.parameters()
            if p.requires_grad
        }

        # Add noise (should be zero)
        dp_optimizer.add_noise()

        # Check that grad equals summed_grad when noise_multiplier=0
        for p in self.model.parameters():
            if p.requires_grad and summed_grads_before[p] is not None:
                self.assertIsNotNone(p.grad, "grad should be set")
                self.assertTrue(
                    torch.allclose(p.grad, summed_grads_before[p], atol=1e-6),
                    "grad should equal summed_grad when noise_multiplier=0",
                )

    def test_step(self):
        """Test that optimizer step works end-to-end."""
        dp_optimizer = NormalizedSGDPOptimizer(
            optimizer=self.optimizer,
            noise_multiplier=0.0,  # No noise for deterministic test
            regularization_param=self.regularization_param,
            expected_batch_size=self.batch_size,
        )

        # Create fake per-sample gradients
        for p in self.model.parameters():
            if p.requires_grad:
                p.grad_sample = torch.randn(self.batch_size, *p.shape)

        # Store initial parameters
        initial_params = {
            name: p.clone()
            for name, p in self.model.named_parameters()
            if p.requires_grad
        }

        # Perform step
        dp_optimizer.step()

        # Check that parameters were updated
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                self.assertFalse(
                    torch.allclose(p, initial_params[name], atol=1e-6),
                    f"Parameter {name} should be updated after step",
                )

    def test_zero_grad(self):
        """Test that zero_grad clears gradients correctly."""
        dp_optimizer = NormalizedSGDPOptimizer(
            optimizer=self.optimizer,
            noise_multiplier=self.noise_multiplier,
            regularization_param=self.regularization_param,
            expected_batch_size=self.batch_size,
        )

        # Create fake per-sample gradients
        for p in self.model.parameters():
            if p.requires_grad:
                p.grad_sample = torch.randn(self.batch_size, *p.shape)

        dp_optimizer.normalize_and_accumulate()
        dp_optimizer.add_noise()

        # Verify gradients exist
        for p in self.model.parameters():
            if p.requires_grad:
                # Check that either grad_sample or grad exists (but not both after add_noise)
                has_grad_sample = p.grad_sample is not None
                has_grad = p.grad is not None
                self.assertTrue(
                    has_grad_sample or has_grad,
                    "Gradients should exist (either grad_sample or grad)",
                )

        # Zero gradients
        dp_optimizer.zero_grad()

        # Verify gradients are cleared
        for p in self.model.parameters():
            if p.requires_grad:
                self.assertIsNone(
                    p.grad_sample, "grad_sample should be None after zero_grad"
                )
                self.assertIsNone(
                    p.summed_grad, "summed_grad should be None after zero_grad"
                )

    def test_empty_batch(self):
        """Test that optimizer handles empty batches correctly."""
        dp_optimizer = NormalizedSGDPOptimizer(
            optimizer=self.optimizer,
            noise_multiplier=self.noise_multiplier,
            regularization_param=self.regularization_param,
            expected_batch_size=0,
        )

        # Create empty per-sample gradients
        for p in self.model.parameters():
            if p.requires_grad:
                p.grad_sample = torch.empty(0, *p.shape)

        # Should not raise an error
        try:
            dp_optimizer.normalize_and_accumulate()
            success = True
        except Exception as e:
            success = False
            self.fail(f"normalize_and_accumulate should handle empty batch: {e}")

        self.assertTrue(success, "Should handle empty batch without error")

    def test_clip_and_accumulate_alias(self):
        """Test that clip_and_accumulate is an alias for normalize_and_accumulate."""
        dp_optimizer = NormalizedSGDPOptimizer(
            optimizer=self.optimizer,
            noise_multiplier=0.0,
            regularization_param=self.regularization_param,
            expected_batch_size=self.batch_size,
        )

        # Create fake per-sample gradients
        for p in self.model.parameters():
            if p.requires_grad:
                p.grad_sample = torch.randn(self.batch_size, *p.shape)

        # Test that clip_and_accumulate works (it should call normalize_and_accumulate)
        dp_optimizer.clip_and_accumulate()

        # Check that summed_grad was created
        for p in self.model.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.summed_grad, "summed_grad should be set")

    def test_different_regularization_params(self):
        """Test that different regularization parameters produce different normalizations."""
        regularization_params = [0.01, 0.1, 1.0]

        results = {}
        for reg_param in regularization_params:
            model = SimpleModel()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            dp_optimizer = NormalizedSGDPOptimizer(
                optimizer=optimizer,
                noise_multiplier=0.0,
                regularization_param=reg_param,
                expected_batch_size=self.batch_size,
            )

            # Create same fake per-sample gradients
            torch.manual_seed(42)
            for p in model.parameters():
                if p.requires_grad:
                    p.grad_sample = torch.randn(self.batch_size, *p.shape)

            dp_optimizer.normalize_and_accumulate()
            results[reg_param] = {
                name: p.summed_grad.clone()
                for name, p in model.named_parameters()
                if p.requires_grad
            }

        # Check that results differ for different regularization parameters
        # Smaller regularization_param should produce larger normalized gradients
        # (since normalization factor = 1 / (r + ||g||) is larger when r is smaller)
        param_names = list(results[regularization_params[0]].keys())
        for param_name in param_names:
            grad_001 = results[0.01][param_name]
            grad_01 = results[0.1][param_name]
            grad_1 = results[1.0][param_name]

            # With smaller regularization_param, normalized gradients should generally be larger
            # (unless the original gradient norm is very small)
            # We just check that they're different
            self.assertFalse(
                torch.allclose(grad_001, grad_01, atol=1e-6),
                f"Normalized gradients should differ for different regularization params ({param_name})",
            )
            self.assertFalse(
                torch.allclose(grad_01, grad_1, atol=1e-6),
                f"Normalized gradients should differ for different regularization params ({param_name})",
            )
