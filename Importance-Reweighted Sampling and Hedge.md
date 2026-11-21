# Importance-Reweighted Sampling and Hedge

These two ideas combine into the standard approach for learning from partial feedback (bandit feedback) in sequential learning.

## 1. The Sequential Learning Setup

At each round $t = 1, \dots, T$:

1. The learner picks a distribution $p_t \in \Delta_K$ over $K$ actions (experts or arms).
2. The adversary picks a loss vector $\ell_t \in [0,1]^K$.
3. The learner samples $I_t \sim p_t$ and suffers loss $\ell_{t, I_t}$.

The goal is low **regret** against the best fixed action in hindsight:

$$
R_T = \sum_{t=1}^{T} \ell_{t, I_t} - \min_{i \in [K]} \sum_{t=1}^{T} \ell_{t,i}
$$

## 2. Hedge (Full Information)

If the learner observes the **entire** loss vector $\ell_t$ every round, Hedge (exponential weights) proceeds as follows.

Maintain cumulative losses:

$$
L_{t,i} = \sum_{s=1}^{t} \ell_{s,i}
$$

Play:

$$
p_{t,i} = \frac{\exp\left(-\eta L_{t-1,i}\right)}{\sum_{j=1}^{K} \exp\left(-\eta L_{t-1,j}\right)}
$$

Actions with larger accumulated loss are exponentially down-weighted. With learning rate $\eta \approx \sqrt{\ln K / T}$:

$$
\mathbb{E}[R_T] = O\left(\sqrt{T \ln K}\right)
$$

## 3. The Problem: Only One Loss Is Observed

In the bandit setting, the learner only observes $\ell_{t, I_t}$, the loss of the action actually played. The losses of the other actions are unknown, so plain Hedge cannot update their weights.

## 4. The Importance-Reweighted Estimator

Build a loss estimate for **every** action from the single observation:

$$
\hat{\ell}_{t,i} = \frac{\ell_{t,i} \, \mathbb{1}\{I_t = i\}}{p_{t,i}}
$$

**Why it works: it is unbiased.**

$$
\mathbb{E}_{I_t \sim p_t}\left[\hat{\ell}_{t,i}\right]
= p_{t,i} \cdot \frac{\ell_{t,i}}{p_{t,i}} + (1 - p_{t,i}) \cdot 0
= \ell_{t,i}
$$

Rarely played actions (small $p_{t,i}$) receive a **large** correction when they are finally sampled, compensating for how seldom they are observed. This is the same idea as importance sampling in Monte Carlo methods: samples drawn from the wrong distribution are reweighted by the ratio of probabilities.

## 5. Putting Them Together: Exp3

Feed the importance-weighted estimates into Hedge:

$$
\hat{L}_{t,i} = \sum_{s=1}^{t} \hat{\ell}_{s,i},
\qquad
p_{t,i} = \frac{\exp\left(-\eta \hat{L}_{t-1,i}\right)}{\sum_{j=1}^{K} \exp\left(-\eta \hat{L}_{t-1,j}\right)}
$$

### Algorithm

```text
Input: learning rate eta, number of actions K
Initialize: L_hat[i] = 0 for all i

for t = 1, ..., T:
    p_t[i] = exp(-eta * L_hat[i]) / sum_j exp(-eta * L_hat[j])
    sample I_t ~ p_t
    observe loss l_t[I_t]
    L_hat[I_t] += l_t[I_t] / p_t[I_t]   # all other coordinates unchanged
```

This is **Exp3** (Exponential-weight algorithm for Exploration and Exploitation). With $\eta \approx \sqrt{\ln K / (T K)}$:

$$
\mathbb{E}[R_T] = O\left(\sqrt{T K \ln K}\right)
$$

The extra $\sqrt{K}$ compared with Hedge is the price of observing one loss per round instead of $K$.

## 6. The Catch: Variance

The estimator is unbiased but can have very large variance. Its second moment is

$$
\mathbb{E}\left[\hat{\ell}_{t,i}^{\,2}\right] = \frac{\ell_{t,i}^{2}}{p_{t,i}} \le \frac{1}{p_{t,i}}
$$

so if $p_{t,i}$ becomes tiny, the estimates blow up. In the analysis of Exp3, the variance term appears as

$$
\sum_{i=1}^{K} p_{t,i} \cdot \mathbb{E}\left[\hat{\ell}_{t,i}^{\,2}\right] \le \sum_{i=1}^{K} p_{t,i} \cdot \frac{1}{p_{t,i}} = K
$$

which is the source of the extra factor $K$ inside the square root.

Common variants that address this:

- **Exp3 with explicit exploration:** mix in uniform noise so that $p_{t,i} \ge \gamma / K$.
- **Exp3-IX:** use the biased estimator $\hat{\ell}_{t,i} = \dfrac{\ell_{t,i} \mathbb{1}\{I_t = i\}}{p_{t,i} + \gamma}$ for better high-probability bounds.
- **Exp3.P:** add confidence-bonus terms for high-probability guarantees.

## Summary

| | Full information | Bandit feedback |
|---|---|---|
| Observation | whole vector $\ell_t$ | only $\ell_{t,I_t}$ |
| Loss used | $\ell_t$ | importance-weighted $\hat{\ell}_t$ |
| Algorithm | Hedge | Exp3 (Hedge + IW) |
| Regret | $O(\sqrt{T \ln K})$ | $O(\sqrt{T K \ln K})$ |

> **Takeaway:** Hedge decides *how to weight* actions based on losses, and importance reweighting supplies *unbiased losses* when you can only observe the actions you sampled.
