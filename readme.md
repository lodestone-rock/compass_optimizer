# Compass optimizer

A modification of original ADAMW optimizer by replacing momentum moment with smoothing filter.

```
Initialize:
θ₀ ∈ ℝᵈ (initial parameters)
m₀ ← 0 (initialize 1st moment vector)
v₀ ← 0 (initialize 2nd moment vector)
t ← 0 (initialize timestep)

Hyperparameters:
α (learning rate)
β₁, β₂ ∈ [0,1) (exponential decay rates for the moment estimates)
ε (small constant to prevent division by zero)

Repeat until convergence:
t ← t + 1
gₜ ← ∇θ fₜ(θₜ₋₁) (compute gradients of the stochastic objective at timestep t)
mₜ ← β₁ mₜ₋₁ + (1 - β₁) gₜ (update biased first moment estimate)
vₜ ← β₂ vₜ₋₁ + (1 - β₂) gₜ² (update biased second moment estimate)
m̂ₜ ← mₜ / (1 - β₁ᵗ) (compute bias-corrected first moment estimate)
v̂ₜ ← vₜ / (1 - β₂ᵗ) (compute bias-corrected second moment estimate)
θₜ ← θₜ₋₁ - α (m̂ₜ / (sqrt(v̂ₜ) + ε)) (update parameters)
```