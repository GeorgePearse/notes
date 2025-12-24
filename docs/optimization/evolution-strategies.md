# Evolution Strategies: From Fundamentals to Hyperscale

A comprehensive guide to evolution strategies for neural network optimization, from basic ES to the EGGROLL algorithm that scales to billion-parameter models.

## Why Evolution Strategies?

ES is a **gradient-free** optimization method. Instead of computing ∂L/∂θ via backprop, you:
1. Sample perturbations around current parameters
2. Evaluate fitness of each perturbation
3. Update parameters toward better-performing samples

```
Backprop:                          Evolution Strategies:
┌─────────────┐                    ┌─────────────┐
│  Forward    │                    │  Sample N   │
│   Pass      │                    │ Perturbations│
└──────┬──────┘                    └──────┬──────┘
       │                                  │
       ▼                                  ▼
┌─────────────┐                    ┌─────────────┐
│  Compute    │                    │  Evaluate   │
│    Loss     │                    │   Fitness   │
└──────┬──────┘                    └──────┬──────┘
       │                                  │
       ▼                                  ▼
┌─────────────┐                    ┌─────────────┐
│  Backward   │                    │   Rank &    │
│   Pass      │                    │   Weight    │
└──────┬──────┘                    └──────┬──────┘
       │                                  │
       ▼                                  ▼
┌─────────────┐                    ┌─────────────┐
│  Update θ   │                    │  Update θ   │
│  via SGD    │                    │  via ES     │
└─────────────┘                    └─────────────┘
```

### When ES Makes Sense

| Use Case | Why ES Works |
|----------|--------------|
| Non-differentiable objectives | Reward functions, discrete outputs, external simulators |
| Sparse/delayed rewards | RL environments with long horizons |
| Parallelization | Each population member evaluates independently |
| Hardware constraints | No backward pass = less memory, simpler kernels |
| Avoiding local optima | Population-based search explores more broadly |

---

## The Core ES Algorithms

### Simple Gaussian ES

The most basic form. Model search distribution as isotropic Gaussian:

```
θ = (μ, σ)    # Mean and step size
```

**Algorithm:**
```python
for generation in range(max_generations):
    # 1. Sample population
    offspring = [μ + σ * np.random.randn(d) for _ in range(λ)]

    # 2. Evaluate fitness
    fitness = [evaluate(x) for x in offspring]

    # 3. Select elite (top-k)
    elite_idx = np.argsort(fitness)[-μ_elite:]
    elite = [offspring[i] for i in elite_idx]

    # 4. Update parameters
    μ = np.mean(elite, axis=0)
    σ = np.std(elite)  # or keep fixed
```

**Problem:** Treats all dimensions equally. Ignores parameter correlations.

---

### CMA-ES (Covariance Matrix Adaptation)

The gold standard for black-box optimization. Tracks full covariance structure.

```
θ = (μ, σ, C)    # Mean, step size, covariance matrix
```

**Key Innovations:**

1. **Covariance Matrix C** - Learns parameter correlations
2. **Evolution Paths** - Accumulates movement history for adaptation
3. **Separate σ Control** - Step size adapts independently from direction

```
Isotropic (Simple ES):              Full Covariance (CMA-ES):
        ○                                   ⬮
       /│\                                 ╱   ╲
      ○ ○ ○                              ○       ○
     Spherical                         Ellipsoidal
     sampling                          sampling
```

**Evolution Paths:**
```python
# Accumulate movement direction
p_σ = (1 - c_σ) * p_σ + sqrt(c_σ * (2 - c_σ) * μ_eff) * C^(-1/2) * (μ_new - μ_old) / σ
p_c = (1 - c_c) * p_c + sqrt(c_c * (2 - c_c) * μ_eff) * (μ_new - μ_old) / σ

# Update step size based on path length
σ = σ * exp((c_σ / d_σ) * (||p_σ|| / E[||N(0,I)||] - 1))

# Update covariance via rank-1 and rank-μ updates
C = (1 - c_1 - c_μ) * C + c_1 * p_c * p_c^T + c_μ * Σ w_i * y_i * y_i^T
```

**Complexity:** O(d²) storage and O(d³) per update for d parameters. Not practical for neural networks with millions of parameters.

---

### Natural Evolution Strategies (NES)

Uses **natural gradients** instead of vanilla gradients:

```
∇̃_θ J(θ) = F_θ^(-1) ∇_θ J(θ)
```

Where F_θ is the Fisher Information Matrix.

**Why Natural Gradients?**

The Fisher matrix measures curvature in probability space, not parameter space:
- Euclidean distance in θ doesn't reflect distribution similarity
- Natural gradient follows steepest descent on the distribution manifold
- Updates are invariant to parameterization

```python
# NES gradient estimate
∇_θ J(θ) ≈ (1/λσ) Σ F(θ + σε_i) * ε_i

# With fitness shaping (rank-based)
utilities = rank_transform(fitness)  # Map to uniform [-0.5, 0.5]
∇_θ J(θ) ≈ (1/λσ) Σ utilities[i] * ε_i
```

---

### OpenAI ES (2017)

Scaled NES to RL with massive parallelism:

**Key Insight:** Only share random seeds, not gradients

```python
# Worker i:
np.random.seed(seed_i)
ε_i = np.random.randn(d)
θ_i = θ + σ * ε_i
reward_i = rollout(θ_i)
send(seed_i, reward_i)  # Tiny communication!

# Master:
for seed_i, reward_i in receive_all():
    np.random.seed(seed_i)
    ε_i = np.random.randn(d)  # Reconstruct perturbation
    gradient += reward_i * ε_i
```

**Additional Tricks:**
- **Mirror sampling:** For each ε, also try -ε (variance reduction)
- **Virtual batch normalization:** Use fixed batch stats across population
- **Fitness shaping:** Rank-based utilities, not raw rewards

**Results:** Solved Atari and MuJoCo using 1440 CPUs, competitive with A3C.

---

## The Scalability Problem

Standard ES has a fatal flaw for large neural networks:

```
Memory per layer:  O(m × n)     # Full perturbation matrix
Population of 1000: O(1000 × m × n)  # Billions of parameters
```

For a 1B parameter model with population 1000:
- Need to store 1000 × 1B = 1 trillion floats
- 4TB of memory just for perturbations

---

## EGGROLL: Evolution Strategies at Hyperscale

The EGGROLL algorithm solves this with **low-rank perturbations**.

### Core Idea

Instead of full-rank perturbation **E** ∈ ℝ^(m×n):

```
E = A B^T    where A ∈ ℝ^(m×r), B ∈ ℝ^(n×r), r ≪ min(m,n)
```

```
Full-rank (standard ES):           Low-rank (EGGROLL):
┌─────────────────┐                ┌───┐   ┌───┐^T
│                 │                │   │   │   │
│    E (m×n)      │      =         │ A │ × │ B │
│                 │                │   │   │   │
│   O(mn) memory  │                │m×r│   │n×r│
└─────────────────┘                └───┘   └───┘
                                   O(r(m+n)) memory
```

### Memory & Compute Savings

| Operation | Full-Rank ES | EGGROLL |
|-----------|--------------|---------|
| Memory per layer | O(mn) | O(r(m+n)) |
| Forward pass cost | O(mn) | O(r(m+n)) |
| Total for population λ | O(λmn) | O(λr(m+n)) |

For m = n = 4096, r = 64:
- Full-rank: 16M parameters per perturbation
- Low-rank: 512K parameters per perturbation
- **32x reduction**

### Convergence Guarantee

EGGROLL converges to full-rank ES updates at rate **O(1/r)**:

```
||∇_full - ∇_lowrank|| = O(1/r)
```

So increasing rank gives predictable improvement. In practice, r = 64-256 works well.

### Algorithm

```python
def eggroll_step(θ, fitness_fn, population_size, rank):
    gradients = []

    for i in range(population_size):
        for layer in θ.layers:
            m, n = layer.weight.shape

            # Low-rank perturbation
            A = randn(m, rank)
            B = randn(n, rank)

            # Perturb: W' = W + σ * A @ B.T
            layer.weight += σ * (A @ B.T)

        # Evaluate
        reward = fitness_fn(θ)

        # Accumulate gradient (reconstruct from A, B)
        gradients.append((reward, A, B))

        # Restore weights
        # ...

    # Update using weighted low-rank gradients
    for layer in θ.layers:
        grad = sum(r * (A @ B.T) for r, A, B in gradients)
        layer.weight += lr * grad
```

### Results

| Metric | Achievement |
|--------|-------------|
| Training speedup | 100x for billion-parameter models |
| Population size | 262,144 (vs ~1000 typical) |
| Reasoning tasks | Outperformed GRPO on GSM8K |
| Throughput | Approaches batched inference speed |

---

## EGG: The Integer-Only RNN

EGGROLL enabled a novel architecture: **Evolved Generative GRU**

### The Insight

If you're not doing backprop, you don't need differentiable activations. So why not:
- Use **int8** throughout (no floats at all)
- Use **integer overflow** as the nonlinearity

```
Standard GRU:                      EGG (Evolved):
┌────────────────────┐             ┌────────────────────┐
│ float32 matmul     │             │ int8 matmul        │
│ sigmoid activation │             │ overflow wrapping  │
│ tanh activation    │             │ (no explicit act)  │
│ float32 state      │             │ int8 state         │
└────────────────────┘             └────────────────────┘
        ↓                                  ↓
   Backprop OK                      No gradients needed
   Slow, high memory                Fast, low memory
```

### Why This Works

Integer overflow in int8 wraps around: 127 + 1 = -128

This creates a sawtooth-like nonlinearity for free:
```
      127 ─┐
          ├┐
          │├┐
          ││├────
   ───────┘││
          ─┘│
     -128 ──┘
```

### Performance

- **10 million tokens/second** on single H100
- Pure int8 = maximum hardware utilization
- No float conversions, no activation kernels

---

## ES vs SGD: When to Use What

| Factor | SGD + Backprop | Evolution Strategies |
|--------|----------------|---------------------|
| **Gradient availability** | Required | Not needed |
| **Sample efficiency** | Better | Worse (needs population) |
| **Parallelization** | Gradient sync needed | Embarrassingly parallel |
| **Memory** | Activations + gradients | Only forward pass |
| **Local optima** | Can get stuck | Population explores broadly |
| **Non-differentiable** | Tricks needed (STE, etc.) | Natural fit |
| **Hardware** | Needs backward kernels | Forward-only |

### The Hybrid Future

Modern approaches combine both:
- **CEM-RL:** Population-based exploration + gradient-based critics
- **PBT:** Evolutionary hyperparameter search + gradient training
- **EGGROLL + GRPO:** ES for reasoning, gradients for pretraining

---

## Exploration Variants

### Novelty-Search ES

Instead of optimizing reward, optimize for **behavioral diversity**:

```python
def novelty(behavior, archive):
    # k-nearest neighbor distance in behavior space
    distances = [dist(behavior, b) for b in archive]
    return np.mean(sorted(distances)[:k])

# Fitness becomes novelty score
fitness = novelty(behavior(θ), archive)
archive.add(behavior(θ))
```

Prevents convergence to single local optimum. Variants:
- **NSR-ES:** Blend novelty + reward
- **NSRA-ES:** Adaptive blending based on progress

### Quality-Diversity

Maintain diverse archive of high-performing solutions:
- MAP-Elites discretizes behavior space into cells
- Each cell keeps best solution for that behavior
- Evolution explores all cells simultaneously

---

## Practical Considerations

### Hyperparameters

| Parameter | Typical Range | Notes |
|-----------|---------------|-------|
| Population size λ | 100-10,000 | Larger = more stable, slower |
| Step size σ | 0.01-0.1 | Too large = chaos, too small = slow |
| Learning rate | 0.001-0.01 | For parameter updates |
| Rank r (EGGROLL) | 32-256 | Trade compute for accuracy |
| Elite ratio | 0.1-0.5 | Fraction selected |

### Fitness Shaping

Raw rewards have high variance. Rank-based transformation helps:

```python
def fitness_shaping(rewards):
    n = len(rewards)
    ranks = np.argsort(np.argsort(rewards))
    utilities = (ranks - n/2) / n  # Center around 0
    return utilities
```

### Implementation Tips

1. **Always use mirror sampling** - halves variance for free
2. **Virtual batch norm** - prevents population collapse
3. **Careful seeding** - enables gradient reconstruction from seeds
4. **Fitness baselines** - subtract running mean for stability

---

## References

- [Evolution Strategies Overview (Lilian Weng)](https://lilianweng.github.io/posts/2019-09-05-evolution-strategies/)
- [Evolution Strategies at Hyperscale (arXiv:2511.16652)](https://arxiv.org/abs/2511.16652)
- [EGGROLL Project Page](https://eshyperscale.github.io/)
- [OpenAI ES Paper (2017)](https://arxiv.org/abs/1703.03864)
- [CMA-ES Tutorial](https://arxiv.org/abs/1604.00772)
