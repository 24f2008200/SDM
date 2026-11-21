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


----

# Worked Example: Exp3 with $K=3$ actions over 4 rounds

## Setup

- Actions: $i \in \{1,2,3\}$, learning rate $\eta = 0.5$, initial $\hat L_{0} = (0,0,0)$.
- The adversary's true losses are below. The learner never sees the full rows, only the entry for the action it plays.
- The sampled actions $I_t$ are fixed in advance, as if they came from the random draws.

| $t$ | $\ell_{t,1}$ | $\ell_{t,2}$ | $\ell_{t,3}$ | sampled $I_t$ |
|---|---|---|---|---|
| 1 | 0.2 | 0.8 | 0.5 | 2 |
| 2 | 0.9 | 0.1 | 0.6 | 1 |
| 3 | 0.3 | 0.7 | 0.4 | 3 |
| 4 | 0.4 | 0.2 | 0.9 | 2 |

True cumulative losses: $(1.8,\ 1.8,\ 2.4)$, so the best fixed action has loss $1.8$.

**Each round:**

1. $p_{t,i} = \dfrac{e^{-\eta \hat L_{t-1,i}}}{\sum_j e^{-\eta \hat L_{t-1,j}}}$
2. Sample $I_t$ and observe $\ell_{t,I_t}$.
3. $\hat\ell_{t,i} = \dfrac{\ell_{t,i}\mathbb{1}\{I_t=i\}}{p_{t,i}}$, so only the played coordinate is nonzero.
4. $\hat L_t = \hat L_{t-1} + \hat\ell_t$.

---

## Round 1

$\hat L_0 = (0,0,0)$, so all weights are $e^0 = 1$:

$$p_1 = \left(\tfrac13,\ \tfrac13,\ \tfrac13\right)$$

- Sampled $I_1 = 2$, observed $\ell_{1,2} = 0.8$. Learner loss: **0.8**.
- Estimate:
$$\hat\ell_{1,2} = \frac{0.8}{1/3} = 2.4, \qquad \hat\ell_1 = (0,\ 2.4,\ 0)$$
- Update: $\hat L_1 = (0,\ 2.4,\ 0)$.

## Round 2

Weights $e^{-0.5\hat L_{1,i}}$:

$$(e^{0},\ e^{-1.2},\ e^{0}) = (1,\ 0.3012,\ 1), \qquad \text{sum} = 2.3012$$

$$p_2 = \left(\tfrac{1}{2.3012},\ \tfrac{0.3012}{2.3012},\ \tfrac{1}{2.3012}\right) = (0.4346,\ 0.1309,\ 0.4346)$$

Action 2 was penalised and is now down-weighted.

- Sampled $I_2 = 1$, observed $\ell_{2,1} = 0.9$. Learner loss: **0.9**.
- Estimate:
$$\hat\ell_{2,1} = \frac{0.9}{0.4346} = 2.071, \qquad \hat\ell_2 = (2.071,\ 0,\ 0)$$
- Update: $\hat L_2 = (2.071,\ 2.4,\ 0)$.

**Unbiasedness check at this round:**

$$\mathbb{E}[\hat\ell_{2,1}] = 0.4346 \cdot \tfrac{0.9}{0.4346} = 0.9 = \ell_{2,1} \checkmark$$

$$\mathbb{E}[\hat\ell_{2,2}] = 0.1309 \cdot \tfrac{0.1}{0.1309} = 0.1 = \ell_{2,2} \checkmark$$

$$\mathbb{E}[\hat\ell_{2,3}] = 0.4346 \cdot \tfrac{0.6}{0.4346} = 0.6 = \ell_{2,3} \checkmark$$

This holds on average over the draw of $I_2$. In any single round the estimate is wrong, because two of its three coordinates are 0.

## Round 3

Weights $e^{-0.5\hat L_{2,i}}$:

$$(e^{-1.0355},\ e^{-1.2},\ e^{0}) = (0.3550,\ 0.3012,\ 1), \qquad \text{sum} = 1.6562$$

$$p_3 = (0.2144,\ 0.1819,\ 0.6038)$$

Action 3 has never been charged, so it now holds most of the probability.

- Sampled $I_3 = 3$, observed $\ell_{3,3} = 0.4$. Learner loss: **0.4**.
- Estimate:
$$\hat\ell_{3,3} = \frac{0.4}{0.6038} = 0.6625, \qquad \hat\ell_3 = (0,\ 0,\ 0.6625)$$
- Update: $\hat L_3 = (2.071,\ 2.4,\ 0.6625)$.

## Round 4

Weights $e^{-0.5\hat L_{3,i}}$:

$$(e^{-1.0355},\ e^{-1.2},\ e^{-0.3313}) = (0.3550,\ 0.3012,\ 0.7180), \qquad \text{sum} = 1.3743$$

$$p_4 = (0.2583,\ 0.2192,\ 0.5225)$$

- Sampled $I_4 = 2$, observed $\ell_{4,2} = 0.2$. Learner loss: **0.2**.
- Estimate:
$$\hat\ell_{4,2} = \frac{0.2}{0.2192} = 0.9125, \qquad \hat\ell_4 = (0,\ 0.9125,\ 0)$$
- Update: $\hat L_4 = (2.071,\ 3.3125,\ 0.6625)$.

---

## Summary of the run

| $t$ | $p_t$ | $I_t$ | observed loss | $\hat\ell_t$ |
|---|---|---|---|---|
| 1 | (0.333, 0.333, 0.333) | 2 | 0.8 | (0, 2.400, 0) |
| 2 | (0.435, 0.131, 0.435) | 1 | 0.9 | (2.071, 0, 0) |
| 3 | (0.214, 0.182, 0.604) | 3 | 0.4 | (0, 0, 0.663) |
| 4 | (0.258, 0.219, 0.522) | 2 | 0.2 | (0, 0.913, 0) |

**Regret on this run:**

$$R_4 = (0.8 + 0.9 + 0.4 + 0.2) - \min(1.8,\ 1.8,\ 2.4) = 2.3 - 1.8 = 0.5$$

## What this example shows

1. **Only one coordinate updates per round.** The importance weight $1/p_{t,I_t}$ scales that one observation up, so each round carries more information about the played action.
2. **A low-probability action gets a bigger jump when sampled.** In round 1 the loss $0.8$ became $2.4$ because $p = 1/3$. If $p$ had been $0.05$, it would have become $16$. This is the variance problem.
3. **A single run is noisy.** After 4 rounds $\hat L_4 = (2.07,\ 3.31,\ 0.66)$, while the true cumulative losses are $(1.8,\ 1.8,\ 2.4)$. The estimates only match the truth on average over many runs. Action 3 looks best because it was sampled once and never penalised, even though it is actually the worst.
4. **Hedge would behave differently.** With full information it would have seen all three losses each round, and $L_4 = (1.8, 1.8, 2.4)$ would have pushed weight away from action 3 immediately.

If you'd like, I can append this example to the GitHub `.md` file, or rerun it with a different $\eta$ or different sampled actions.


--------

# Hedge vs. Exp3: What's Shared and What's Different

Your picture is right, with two refinements:

- The scaling is by **$1/p$**, not $p$. That is what makes $\langle p_t, \hat\ell_t\rangle$ equal the loss you actually suffered:
$$\langle p_t, \hat\ell_t\rangle = \sum_i p_{t,i}\cdot\frac{\ell_{t,i}\mathbb{1}\{I_t=i\}}{p_{t,i}} = \ell_{t,I_t}$$
- Exp3 also updates **every** round. The unplayed actions get an update of $0$, and that zero still matters, as explained below.

## Common ideas

1. **Both are exponential weights on cumulative loss.** The rule $p_{t,i} \propto e^{-\eta \cdot (\text{cumulative loss of } i)}$ is identical. Exp3 is literally Hedge run on a different loss sequence, $\hat\ell$ in place of $\ell$.
2. **Both are "follow the leader, but softened".** Pure follow-the-leader would put all weight on the current best action, and an adversary can exploit that. The exponential smooths it, and $\eta$ controls how sharp it is.
3. **Both rely on a ledger.** Neither algorithm models the world. Each just keeps a running score per action and converts scores into probabilities.
4. **Both are analysed with the same potential-function argument.** Each has a "learning" term $\ln K/\eta$ and a "stability" term proportional to $\eta$.

## Where they differ (beyond full vs. partial)

**1. Exact ledger vs. noisy ledger**

In Hedge, $L_{t,i}$ is the true cumulative loss. In Exp3, $\hat L_{t,i}$ is a random quantity that is correct only in expectation. In your 4-step run, $\hat L_4=(2.07, 3.31, 0.66)$ against true losses of $(1.8, 1.8, 2.4)$. Hedge never ranks an action wrongly because of bad luck, but Exp3 can.

**2. Your actions change what you learn**

In Hedge, $p_t$ affects only your own loss, not your information. In Exp3 there is a feedback loop:

$$p_t \to \text{what you sample} \to \hat\ell_t \to \hat L_t \to p_{t+1}$$

This is the exploration vs. exploitation tension. If $p_{t,i}$ is small you rarely see action $i$, so you learn about it slowly. Hedge has no such tension.

**3. Not being observed acts like optimism**

Losses are nonnegative, so $\hat L$ never decreases. An unplayed action's score stays frozen while the played action's score rises. Relative to the others, the frozen action looks better and its probability climbs back up. This works as built-in exploration: no news about an action is treated as good news. It is what stops a neglected action from being permanently abandoned.

**4. Jumps are uneven in size and timing**

Hedge makes small, steady updates to every action. Exp3 makes rare, large updates to one action, of size $\ell/p$. Both have the same expected total. Hedge's path is smooth, while Exp3's is jerky, and that jerkiness is variance.

**5. The stability term, which is where $\sqrt K$ comes from**

The regret analysis has a term of the form

$$\eta \sum_t \mathbb{E}\Big[\textstyle\sum_i p_{t,i}\,(\text{loss used})_{t,i}^2\Big]$$

| | Hedge | Exp3 |
|---|---|---|
| Loss used | $\ell_{t,i}\le 1$ | $\hat\ell_{t,i}$ |
| Per-round term | $\sum_i p_{t,i}\ell_{t,i}^2 \le 1$ | $\sum_i p_{t,i}\cdot\frac{\ell_{t,i}^2}{p_{t,i}} \le K$ |
| Best $\eta$ | $\sqrt{\ln K/T}$ | $\sqrt{\ln K/(TK)}$ |
| Regret | $\sqrt{T\ln K}$ | $\sqrt{TK\ln K}$ |

The importance weight $1/p$ exactly cancels the $p$ in front of it, so each action contributes $1$ and the $K$ actions sum to $K$. That is the entire $\sqrt K$ penalty. Because the noise is larger, Exp3 must also learn more slowly (smaller $\eta$) to stay stable.

**6. Guarantees hold in a weaker sense**

Hedge's regret bound is for the sequence you actually saw. Exp3's basic bound is in expectation over its own sampling. Getting a high-probability bound needs variants (Exp3-IX, Exp3.P), because a single unlucky tiny-$p$ draw can produce a huge $\hat\ell$.

## One-line intuition

Hedge keeps a **bank statement** that is exact and updated for every action each round. Exp3 keeps an **estimated statement**, updated only where it looked, with each observation inflated by $1/p$ so that the totals are right on average. The decision rule is the same, but the quality of the numbers going into it is different. Almost everything else (smaller $\eta$, the $K$ factor, the need for exploration, the variance fixes) follows from that.
