# Clipping Strategies Usage Guide

## Installation

To use a custom Opacus fork, install directly from the git repository:

```bash
# Install from specific branch/commit
pip install "opacus @ git+https://github.com/parths007/opacus.git@master-thesis"

# Or install in editable mode if you've cloned the repository
git clone https://github.com/parths007/opacus.git
cd opacus
git checkout master-thesis
pip install -e .

# Optional: Install with dev dependencies for development
pip install -e .[dev]
```

Note: The `@ git+...` syntax installs directly from the repository. Use editable mode (`-e`) if you need to modify the code and see changes immediately without reinstalling.

## Overview

Opacus supports multiple gradient clipping strategies for differentially private training. Set the `clipping` parameter in `PrivacyEngine.make_private()`.

---

## Algorithm Implementations with Code Explanations

This section provides detailed step-by-step explanations of how each clipping strategy is implemented in PyTorch, including the mathematical formulas and their corresponding code.

### Standard DP-SGD Clipping (Baseline)

Paper: [Deep Learning with Differential Privacy](https://arxiv.org/pdf/1607.00133) (Abadi et al., 2016)

Before diving into the custom optimizers, it's important to understand standard DP-SGD clipping as the baseline:

Mathematical Formula:
```
clip_factor = min(1, C / ||g_i||)
g'_i = clip_factor × g_i
```

Where:
- `C` is the clipping threshold (max_grad_norm)
- `||g_i||` is the L2 norm of sample i's gradient
- The resulting clipped gradient has norm `||g'_i|| ≤ C`

PyTorch Implementation (`opacus/optimizers/optimizer.py`):
```python
# Step 1: Compute per-parameter L2 norms for each sample
per_param_norms = [g.reshape(len(g), -1).norm(2, dim=-1) for g in self.grad_samples]

# Step 2: Compute per-sample total gradient norm (combine all parameters)
per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)

# Step 3: Compute clip factor with clamp(max=1.0) - key difference from automatic clipping
per_sample_clip_factor = (self.max_grad_norm / (per_sample_norms + 1e-6)).clamp(max=1.0)

# Step 4: Apply clipping and sum gradients across samples
grad = torch.einsum("i,i...", per_sample_clip_factor, grad_sample)
```

Key Characteristic: The `.clamp(max=1.0)` ensures gradients are never scaled UP, only clipped down.

---

### 1. Automatic Clipping (AUTO-S)

Paper: [Automatic Clipping: Differentially Private Deep Learning Made Easier and Stronger](https://arxiv.org/pdf/2206.07136)

Implementation File: `opacus/optimizers/optimizer_automatic_clipping.py`

#### Core Idea

Instead of hard clipping with `min(1, C/||g||)`, automatic clipping uses a normalization-based approach. The paper proposes AUTO-S which normalizes each gradient by its own norm, eliminating the need for threshold tuning.

#### Mathematical Formula

Paper Formula (AUTO-S):
```
g' = g_i / (||g_i|| + γ)
```

Implementation Formula (adapted for Opacus):
```
scaling_factor = C / (||g_i|| + γ)
g' = scaling_factor × g_i
```

Where:
- `C` is the clipping norm (max_grad_norm) - added for compatibility with Opacus noise calibration
- `γ = 0.01` is a stabilization constant to prevent division by zero
- `||g_i||` is the per-sample gradient norm

Note: The implementation multiplies by `C` to maintain the same noise calibration (`σ = noise_multiplier × C`) as standard DP-SGD. This is mathematically equivalent to the paper's formulation when adjusting the noise multiplier accordingly.

#### Step-by-Step Code Explanation

```python
class DPAutomaticClippingOptimizer(DPOptimizer):
    def clip_and_accumulate(self):
        # Step 1: Compute per-parameter norms for each sample
        # For each parameter, reshape gradients to [batch_size, -1] and compute L2 norm
        per_param_norms = [
            g.reshape(len(g), -1).norm(2, dim=-1) for g in self.grad_samples
        ]
        
        # Step 2: Move all norms to same device for stacking
        target_device = per_param_norms[0].device
        per_param_norms = [norm.to(target_device) for norm in per_param_norms]
        
        # Step 3: Compute total per-sample gradient norm (L2 across all parameters)
        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
        
        # Step 4: KEY DIFFERENCE - Automatic clipping factor WITHOUT clamp
        # Standard: (C / (||g|| + ε)).clamp(max=1.0)
        # Automatic: C / (||g|| + 0.01) - no clamp, uses larger stabilization constant
        per_sample_clip_factor = self.max_grad_norm / (per_sample_norms + 0.01)
        
        # Step 5: Apply scaling to each parameter and accumulate
        for p in self.params:
            grad_sample = self._get_flat_grad_sample(p)
            # Einstein summation: multiply each sample's gradient by its clip factor and sum
            grad = torch.einsum("i,i...", per_sample_clip_factor, grad_sample)
            p.summed_grad = grad if p.summed_grad is None else p.summed_grad + grad
```

#### Key Differences from Standard Clipping

<table>
  <thead>
    <tr>
      <th>Aspect</th>
      <th>Standard Clipping</th>
      <th>Automatic Clipping</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Formula</td>
      <td><code>min(1, C/||g||)</code></td>
      <td><code>C/(||g|| + 0.01)</code></td>
    </tr>
    <tr>
      <td>Clamp</td>
      <td>Yes (max=1.0)</td>
      <td>No</td>
    </tr>
    <tr>
      <td>Small gradients</td>
      <td>Unchanged</td>
      <td>Scaled up</td>
    </tr>
    <tr>
      <td>Large gradients</td>
      <td>Clipped to C</td>
      <td>Scaled down</td>
    </tr>
    <tr>
      <td>Stabilization</td>
      <td>1e-6</td>
      <td>0.01</td>
    </tr>
  </tbody>
</table>

#### Intuition

- Standard clipping: Only reduces large gradients, leaves small ones unchanged
- Automatic clipping: Normalizes all gradients proportionally, making optimization more uniform across samples

---

### 2. Per-Sample Adaptive Clipping (PSAC)

Paper: [Differentially Private Learning with Per-Sample Adaptive Clipping](https://arxiv.org/pdf/2212.00328)

Implementation File: `opacus/optimizers/psacoptimizer.py`

#### Core Idea

Instead of using a fixed clipping threshold `C` for all samples, PSAC computes an adaptive threshold `C_i` for each sample using a non-monotonic weight function. This reduces the deviation between clipped and true gradients.

#### Mathematical Formula (from Paper, Algorithm 1)

Weight Function (Non-Monotonic):

The paper defines the weight function as:
```
w(||g||) = 1 / (||g|| + r/(||g|| + r))
```

Which simplifies algebraically to:
```
w(||g||) = (||g|| + r) / (||g||² + r·||g|| + r)
```

Adaptive Clipping Threshold:
```
C_i = C × w(||g_i||)
```

Final Clipped Gradient:
```
g'_i = g_i × min(1, C_i / ||g_i||)
```

Where:
- `r` is a hyperparameter (default: 0.1, paper suggests 0.01 or 0.1)
- `C` is the global max_grad_norm (clipping bound)
- `||g_i||` is the gradient norm for sample i

#### Step-by-Step Code Explanation

```python
class PSACDPOptimizer(DPOptimizer):
    def __init__(self, ..., r: float = 0.1, ...):
        self.r = r  # Hyperparameter controlling adaptation level
        
    def _compute_per_sample_adaptive_clip_norms(self, per_sample_norms):
        """
        Compute adaptive clipping thresholds using non-monotonic weight function.
        """
        r = self.r
        norms = per_sample_norms
        
        # Step 1: Compute non-monotonic weight function
        # w(||g||) = (||g|| + r) / (||g||² + r·||g|| + r)
        numerator = norms + r
        denominator = norms * norms + r * norms + r
        weight = numerator / denominator
        
        # Step 2: Compute adaptive clipping threshold: C_i = C × w(||g_i||)
        adaptive_clip_norms = self.max_grad_norm * weight
        
        # Step 3: Clamp for numerical stability
        min_clip_norm = self.max_grad_norm * r / (1.0 + r)
        adaptive_clip_norms = torch.clamp(
            adaptive_clip_norms, min=min_clip_norm, max=self.max_grad_norm
        )
        
        return adaptive_clip_norms
    
    def clip_and_accumulate(self):
        # Step 1: Compute per-parameter norms for each sample
        per_param_norms = [
            g.reshape(len(g), -1).norm(2, dim=-1) for g in self.grad_samples
        ]
        
        # Step 2: Compute per-sample gradient norms (L2 across all parameters)
        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
        
        # Step 3: KEY STEP - Compute adaptive clipping thresholds
        self._per_sample_clip_norms = self._compute_per_sample_adaptive_clip_norms(
            per_sample_norms
        )
        
        # Step 4: Compute per-sample clip factors: min(1, C_i / ||g_i||)
        per_sample_clip_factor = (
            self._per_sample_clip_norms / (per_sample_norms + 1e-8)
        ).clamp(max=1.0)
        
        # Step 5: Apply clipping and accumulate
        for p in self.params:
            grad_sample = self._get_flat_grad_sample(p)
            grad = torch.einsum("i,i...", per_sample_clip_factor, grad_sample)
            p.summed_grad = grad if p.summed_grad is None else p.summed_grad + grad
```

#### Weight Function Properties

The non-monotonic weight function `w(||g||) = (||g|| + r) / (||g||² + r·||g|| + r)` has special properties:

<table>
  <thead>
    <tr>
      <th>Gradient Norm</th>
      <th>Weight <code>w</code></th>
      <th>Adaptive Threshold <code>C_i = C × w</code></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Very small (<code>||g|| → 0</code>)</td>
      <td><code>w → 1</code></td>
      <td><code>C_i → C</code> (standard clipping threshold)</td>
    </tr>
    <tr>
      <td>Medium</td>
      <td><code>w</code> decreases from 1</td>
      <td><code>C_i &lt; C</code> (moderate adaptive threshold)</td>
    </tr>
    <tr>
      <td>Very large (<code>||g|| → ∞</code>)</td>
      <td><code>w → 0</code></td>
      <td><code>C_i → 0</code> (aggressive clipping)</td>
    </tr>
  </tbody>
</table>

Why Non-Monotonic? Unlike monotonic functions like `1/(||g|| + r)` which over-weight small gradients, the PSAC weight function:
- Treats small gradients similarly to standard clipping
- Progressively clips larger gradients more aggressively
- Avoids the bias introduced by uniformly scaling all gradients

#### Noise Calibration

```python
def add_noise(self):
    # Noise is calibrated based on max adaptive clip norm
    if self._per_sample_clip_norms is not None:
        max_adaptive_clip_norm = self._per_sample_clip_norms.max().item()
        effective_clip_norm = min(max_adaptive_clip_norm, self.max_grad_norm)
    else:
        effective_clip_norm = self.max_grad_norm
    
    # Add Gaussian noise scaled by effective clipping norm
    noise = _generate_noise(
        std=self.noise_multiplier * effective_clip_norm,
        reference=p.summed_grad,
        ...
    )
```

---

### 3. Normalized SGD with Perturbation (NSGD)

Paper: [Normalized/Clipped SGD with Perturbation for
Differentially Private Non-Convex Optimization](https://arxiv.org/pdf/2206.13033)

Implementation File: `opacus/optimizers/normalized_sgd_optimizer.py`

#### Core Idea

Instead of clipping gradients, NSGD normalizes them by dividing each per-sample gradient by its norm plus a regularization parameter. This ensures all gradients have bounded norm without hard clipping.

#### Mathematical Formula (from Paper)

Per-Sample Gradient Normalization:
```
g'_i = g_i / (r + ||g_i||)
```

Aggregated Gradient with Noise:
```
g_agg = (1/n) × Σ g'_i + N(0, σ²I)
```

Where:
- `r > 0` is the regularization parameter (implementation uses `max_grad_norm`)
- `||g_i||` is the per-sample gradient L2 norm
- `n` is the batch size
- `σ = noise_multiplier × r` (noise calibrated to regularization parameter)

Key Property: After normalization, `||g'_i|| = ||g_i|| / (r + ||g_i||) < 1` for all samples, ensuring bounded sensitivity.

#### Step-by-Step Code Explanation

```python
class NormalizedSGDPOptimizer(DPOptimizer):
    def __init__(self, ..., regularization_param=None, max_grad_norm=None, ...):
        # Use max_grad_norm as regularization_param for compatibility
        if regularization_param is None:
            regularization_param = max_grad_norm
        
        self.regularization_param = regularization_param
        
    def normalize_and_accumulate(self):
        """
        Normalizes gradients instead of clipping.
        """
        # Step 1: Compute per-parameter norms for each sample
        per_param_norms = [
            g.reshape(len(g), -1).norm(2, dim=-1) for g in self.grad_samples
        ]
        
        # Step 2: Compute per-sample gradient norms
        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
        
        # Step 3: KEY DIFFERENCE - Normalization factor instead of clip factor
        # Clipping: min(1, C / ||g||)
        # NSGD:     1 / (r + ||g||)
        per_sample_norm_factor = 1.0 / (per_sample_norms + self.regularization_param)
        
        # Step 4: Apply normalization and accumulate
        for p in self.params:
            grad_sample = self._get_flat_grad_sample(p)
            grad = torch.einsum("i,i...", per_sample_norm_factor, grad_sample)
            p.summed_grad = grad if p.summed_grad is None else p.summed_grad + grad
    
    def clip_and_accumulate(self):
        """Alias for compatibility with DPOptimizer interface."""
        self.normalize_and_accumulate()
    
    def add_noise(self):
        """
        Noise scaled by regularization_param, not max_grad_norm.
        """
        for p in self.params:
            noise = _generate_noise(
                std=self.noise_multiplier * self.regularization_param,
                reference=p.summed_grad,
                ...
            )
            p.grad = (p.summed_grad + noise).view_as(p)
```

#### Comparison with Other Methods

<table>
  <thead>
    <tr>
      <th>Method</th>
      <th>Transformation</th>
      <th>Resulting Norm Bound</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Standard Clipping</td>
      <td><code>g × min(1, C/||g||)</code></td>
      <td><code>||g'|| &le; C</code></td>
    </tr>
    <tr>
      <td>Automatic Clipping (Paper)</td>
      <td><code>g / (||g|| + γ)</code></td>
      <td><code>||g'|| &lt; 1</code></td>
    </tr>
    <tr>
      <td>Automatic Clipping (Impl.)</td>
      <td><code>g × C / (||g|| + γ)</code></td>
      <td><code>||g'|| &lt; C</code></td>
    </tr>
    <tr>
      <td>NSGD</td>
      <td><code>g / (r + ||g||)</code></td>
      <td><code>||g'|| &lt; 1</code></td>
    </tr>
  </tbody>
</table>


#### Why Normalization Works

1. Bounded sensitivity: After normalization, each gradient has norm ≤ 1
2. No information loss: Gradients are scaled proportionally, preserving direction
3. Smooth transformation: No discontinuity at the clipping boundary

---

## Comparison Summary

<table>
  <thead>
    <tr>
      <th>Optimizer</th>
      <th>Paper</th>
      <th>Formula (per-sample)</th>
      <th>Key Benefit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Standard (Flat)</td>
      <td>Abadi et al., 2016</td>
      <td><code>g' = g × min(1, C/||g||)</code></td>
      <td>Simple, well-understood baseline</td>
    </tr>
    <tr>
      <td>Automatic (AUTO-S)</td>
      <td>Bu et al., 2022</td>
      <td><code>g' = g / (||g|| + γ)</code></td>
      <td>No threshold tuning needed</td>
    </tr>
    <tr>
      <td>PSAC</td>
      <td>Xia et al., 2022</td>
      <td><code>g' = g × min(1, C·w(||g||)/||g||)</code></td>
      <td>Reduces clipping bias with adaptive thresholds</td>
    </tr>
    <tr>
      <td>NSGD</td>
      <td>Zhang et al., 2022</td>
      <td><code>g' = g / (r + ||g||)</code></td>
      <td>Bounded sensitivity (||g'|| &lt; 1)</td>
    </tr>
  </tbody>
</table>

Where:
- `C` = clipping threshold (max_grad_norm)
- `γ` = stabilization constant (0.01)
- `r` = regularization parameter  
- `w(||g||) = (||g|| + r) / (||g||² + r·||g|| + r)` (PSAC weight function)

---

## Available Strategies

### 1. Flat Clipping (Default)

```python
model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.0,
    max_grad_norm=1.0,  # Single float value
    clipping="flat",
)
```

### 2. Per-Layer Clipping

```python
# Provide list of max_grad_norm (one per parameter)
max_grad_norm = [1.0, 1.5, 0.8, ...]  # One value per layer
model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.0,
    max_grad_norm=max_grad_norm,
    clipping="per_layer",
)
```

### 3. Automatic Clipping

```python
model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
    clipping="automatic",
)
```

### 4. Automatic Per-Layer Clipping

```python
max_grad_norm = [1.0, 1.5, 0.8, ...]  # One value per layer
model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.0,
    max_grad_norm=max_grad_norm,
    clipping="automatic_per_layer",
)
```

### 5. Adaptive Clipping (AdaClip)

```python
from opacus.optimizers import AdaClipDPOptimizer

# Requires additional parameters
optimizer = AdaClipDPOptimizer(
    optimizer=optimizer,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
    target_unclipped_quantile=0.5,
    clipbound_learning_rate=0.2,
    max_clipbound=1e8,
    min_clipbound=1.0,
    unclipped_num_std=1.0,
    expected_batch_size=batch_size,
)
```

Note: For adaptive clipping with ghost clipping, use `PrivacyEngineAdaptiveClipping`:

```python
from opacus.utils.adaptive_clipping import PrivacyEngineAdaptiveClipping

privacy_engine = PrivacyEngineAdaptiveClipping()
model, optimizer, criterion, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    criterion=criterion,
    noise_multiplier=1.0,
    max_grad_norm=10.0,  # Initial clipping norm
    grad_sample_mode="ghost",
    target_unclipped_quantile=0.5,
    min_clipbound=1.0,
    max_clipbound=1e8,
    clipbound_learning_rate=0.2,
)
```

### 6. Per-Sample Adaptive Clipping (PSAC)

```python
model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
    clipping="psac",
)
```

### 7. Normalized SGD

```python
model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
    clipping="normalized_sgd",
)
```

## Distributed Training

For distributed training, set `distributed=True`:

```python
model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
    clipping="flat",  # or "per_layer", "automatic", etc.
    distributed=True,
)
```

## Ghost Clipping (Fast Gradient Clipping)

Use `grad_sample_mode="ghost"` for memory-efficient training (only with flat clipping):

```python
model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
    clipping="flat",
    grad_sample_mode="ghost",
)
```
