# Compass optimizer

A modification of original ADAMW optimizer by replacing momentum moment with smoothing filter.

```git
Initialize:
θ₀ ∈ ℝᵈ (initial parameters)
m₀ ← 0 (initialize 1st moment vector)
v₀ ← 0 (initialize 2nd moment vector)
t ← 0 (initialize timestep)

Hyperparameters:
α (learning rate)
γ (smoothing factor)
λ (weight decay)
β₁, β₂ ∈ [0,1) (exponential decay rates for the moment estimates)
ε (small constant to prevent division by zero)

Repeat until convergence:
t ← t + 1
gₜ ← ∇θ fₜ(θₜ₋₁) (compute gradients of the stochastic objective at timestep t)
mₜ ← β₁ mₜ₋₁ + (1 - β₁) gₜ (update biased first moment estimate)
ĝₜ ← gₜ + mₜγ (smooth out gradients)
vₜ ← β₂ vₜ₋₁ + (1 - β₂) ĝₜ² (update biased second moment estimate)
m̂ₜ ← mₜ / (1 - β₁ᵗ) (compute bias-corrected first moment estimate)
v̂ₜ ← vₜ / (1 - β₂ᵗ) (compute bias-corrected second moment estimate)
Θₜ₋₁ ← θₜ₋₁(1 - αλ) (compute decoupled weight decay)
θₜ ← Θₜ₋₁ - αĝₜ / (m̂ₜsqrt(v̂ₜ) + ε) (update parameters)
```