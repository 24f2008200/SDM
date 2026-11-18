# Sequential Decision Making — Week 7 Notes
**Part 2 begins: Multi-Armed Bandits (partial information), Importance-Weighted Estimates, EXP3, EXP3-IX**

> Continuation of the Week 1–4, Week 5 and Week 6 notes. Built from lecture transcript L39–45.
> Math is GitHub-flavoured LaTeX (`$...$`, `$$...$$`).
> Notation: $d$ = number of arms, $T$ = rounds, $I_t$ = arm played in round $t$, $\ell_t\in[0,1]^d$ = (hidden) loss vector, $p_t\in\Delta_d$ = sampling distribution, $\hat\ell_t$ = estimated loss vector, $\eta$ = learning rate (written "ea" in the transcript), $\ln$ = natural log.

---

## Table of Contents
1. [One-page summary](#1-one-page-summary)
2. [From full information to partial information](#2-from-full-information-to-partial-information)
3. [Why "bandits"? Applications](#3-why-bandits-applications)
4. [The bandit protocol](#4-the-bandit-protocol)
5. [Adversarial vs stochastic bandits](#5-adversarial-vs-stochastic-bandits)
6. [Oblivious adversary and regret for bandits](#6-oblivious-adversary-and-regret-for-bandits)
7. [The central idea: estimate the loss vector (importance weighting)](#7-the-central-idea-estimate-the-loss-vector-importance-weighting)
8. [The algorithm (EXP3)](#8-the-algorithm-exp3)
9. [Regret analysis of EXP3](#9-regret-analysis-of-exp3)
10. [Why "EXP3"? Exploration and exploitation](#10-why-exp3-exploration-and-exploitation)
11. [The learning rate: bandit vs full information](#11-the-learning-rate-bandit-vs-full-information)
12. [Expected regret vs high-probability regret](#12-expected-regret-vs-high-probability-regret)
13. [EXP3-IX (implicit exploration) and EXP3.P (explicit exploration)](#13-exp3-ix-implicit-exploration-and-exp3p-explicit-exploration)
14. [Where next](#14-where-next)
15. [Formula table](#15-formula-table)
16. [Quiz preparation tips](#16-quiz-preparation-tips)
17. [Self-test questions with answers](#17-self-test-questions-with-answers)
18. [Lecture slips / things to double-check](#18-lecture-slips--things-to-double-check)
19. [Runnable verification code](#19-runnable-verification-code)

---

## 1. One-page summary

| Idea | Statement |
|---|---|
| **Part 2 setting** | Sequential decisions with **partial (bandit) feedback**: you only see the loss of the arm you played |
| **Protocol** | Round $t$: choose arm $I_t\in\{1..d\}$ (randomly, from $p_t$); observe only $\ell_{t,I_t}\in[0,1]$ (not the vector $\ell_t$) |
| **Two flavours** | **Adversarial** (loss matrix fixed in advance by an oblivious adversary) vs **stochastic** (each arm i.i.d. with unknown mean $\theta_i$). Week 7 does adversarial; stochastic comes later |
| **Randomisation is necessary** | Same Cover argument as before: no deterministic algorithm beats a deterministic oblivious adversary |
| **Regret** | $R_T=\mathbb E\big[\sum_t\ell_{t,I_t}\big]-\min_i\sum_t\ell_{t,i}$ (expected, vs best fixed arm, including arms we never played) |
| **Key trick** | Build an **importance-weighted loss estimate** $\hat\ell_{t,i}=\dfrac{\ell_{t,i}}{p_{t,i}}\mathbf 1[I_t=i]$: zero everywhere except the played arm. It is **unbiased**: $\mathbb E[\hat\ell_t\mid p_t]=\ell_t$ |
| **EXP3** | Run **Hedge on $\hat\ell_t$**: $w_{t+1,i}=w_{t,i}e^{-\eta\hat\ell_{t,i}}$, $p_t=w_t/\sum_jw_{t,j}$ |
| **Obstacle** | $\hat\ell_t$ is unbounded ($1/p_{t,i}$ can be huge), so Hedge's "$\lVert\ell\rVert^2\le d$" step fails |
| **Fix** | Don't bound $p$ away: $\mathbb E\big[\langle p_t,\hat\ell_t^2\rangle\big]=\sum_i\ell_{t,i}^2\le d$ (the $p$'s cancel) |
| **Result** | $R_T\le\dfrac{\ln d}{\eta}+\dfrac{\eta Td}{2}$; $\eta^*=\sqrt{\dfrac{2\ln d}{Td}}$; $\ \mathbf{R_T\le\sqrt{2\,T\,d\,\ln d}}=O(\sqrt{Td\ln d})$ |
| **Cost vs Hedge** | Extra factor $\sqrt d$ (and a learning rate $\sqrt d$ times smaller): you can't trust an estimate as much as the truth |
| **Name** | **EXP3** = **Exp**onentiation, **Exp**loration, **Exp**loitation (Auer et al., 2002) |
| **Limitation** | Bound holds only **in expectation**; no high-probability guarantee |
| **EXP3-IX** (2015) | Divide by $p_{t,i}+\gamma$ instead of $p_{t,i}$ ("implicit exploration") $\Rightarrow$ high-probability regret $O(\sqrt{Td\ln d})$ |
| **EXP3.P** | Explicit exploration: $p_{t,i}=(1-\gamma)\frac{w_{t,i}}{\sum_jw_{t,j}}+\frac\gamma d$ (plus a modified loss estimate) |

**Five ideas to remember**
1. **Bandit = learning from a single coordinate.** To reuse Hedge we need a full loss vector, so we **manufacture** one.
2. **Importance weighting** (divide by the probability of being observed) makes a one-coordinate observation an **unbiased** estimate of the whole vector.
3. The analysis reuses Hedge's inequality; the only new step is bounding $\mathbb E\langle p_t,\hat\ell_t^2\rangle$, and the probabilities **cancel** exactly.
4. The price of partial information is a **$\sqrt d$** factor in regret (the rate in $T$ is still $\sqrt T$).
5. EXP3 balances **exploration/exploitation implicitly**: pulling an arm with positive loss lowers its weight and raises everyone else's.

---

## 2. From full information to partial information

**Part 1 (done):** full information. You choose an action but see the feedback of *all* actions (stock experts: you learn how every expert performed), which lets you up/down-weight every expert (Hedge).

**Part 2:** **partial information.** You only see the feedback for the action you took.
* Medical example: you prescribe one medication and observe whether it helped; you do **not** learn what would have happened with another medication.
* Harder than full information, but the ideas from Part 1 (Hedge, regret, regularisation) will be the building blocks.

The standard name for this class of problems: **multi-armed bandits**.

---

## 3. Why "bandits"? Applications

**Origin of the name.** Casino slot machines ("one-armed bandits") have levers (arms). You pay, pull an arm, and get a reward (possibly less than you paid, rarely a jackpot). Casinos are designed so that on average the house wins, so over time the player's money is "stolen": hence *bandits*. Here: $d$ arms $A_1,\dots,A_d$, one pull per round.

**Applications** (arm = the thing you choose; feedback = outcome for that choice only):

| Application | Arms | Feedback |
|---|---|---|
| Medical treatment / vaccination | medications | did the patient improve? |
| Online advertising | ad designs | did the user click? (click-through rate $\to$ possible sale) |
| Recommendation (Netflix/Prime) | movies to show in a **slate** (a few slots among thousands of titles) | did the user click / stay on the platform? |
| Casino | levers | payout |

In each case you cannot know what would have happened with an alternative choice (a different ad for the same user, ...).

---

## 4. The bandit protocol

For $t=1,\dots,T$ (or $\infty$):
1. The algorithm chooses an arm $I_t\in\{1,\dots,d\}$.
2. It **observes (and suffers) only** the loss $\ell_{t,I_t}\in[0,1]$.

$$
\ell_t=(\ell_{t,1},\dots,\ell_{t,d})\in[0,1]^d\ \text{ exists in round }t\text{ but is NOT revealed; only the coordinate }I_t\text{ is.}
$$

**Key difference from online learning:** in Hedge you observed the whole vector $\ell_t$; here only $\ell_{t,I_t}$. The whole of Part 2 is about coping with this.

---

## 5. Adversarial vs stochastic bandits

Two ways the losses/rewards can be generated:

| | Adversarial bandits | Stochastic bandits |
|---|---|---|
| Source of feedback | an adversary tries to make you suffer (as in Part 1) | a **fixed random process**: arm $i$ has an unknown mean $\theta_i$ (a larger $\theta_i$ = better arm) and each pull returns an i.i.d. sample with expectation $\theta_i$ |
| Example | worst-case sequences | medication $i$ cures with unknown probability $\theta_i$ (a biased coin: heads = cured) |
| Goal | small regret vs best fixed arm in hindsight | find the arm with the largest $\theta_i$ |
| Difficulty | harder (no statistical regularity) | seemingly easier (same distribution each time, so you can learn $\theta_i$ statistically) |

**Plan of Part 2:** first **adversarial** bandits (this week; reuses Part 1's tools), later **stochastic** bandits (different algorithms).

---

## 6. Oblivious adversary and regret for bandits

### 6.1 Deterministic oblivious adversary
* The adversary **knows the algorithm's pseudo-code** but **fixes the entire loss matrix before the game begins** (rounds $t=1,2,\dots$; arms $1..d$; all entries in $[0,1]$).
* Each round only the entry for the arm you played is revealed (e.g. you play arm 3 in round 1: you see $\ell_{1,3}$ only; arm 1 in round 2: you see $\ell_{2,1}$ only).
* It cannot adapt to which arms you actually played (it cannot "bump up arm 2's losses because you never played arm 2").

### 6.2 Randomisation is a must
By Cover's argument (Week 2–4): **no deterministic algorithm** can beat a deterministic oblivious adversary (the adversary can simulate the algorithm). So the algorithm must be **randomised**: in round $t$ it keeps a distribution $p_t\in\Delta_d$ over arms and samples $I_t\sim p_t$.

Note that $p_t$ is itself **random**, since it depends on the previous draws $I_1,\dots,I_{t-1}$.

### 6.3 Regret
$$
\boxed{\ R_T=\mathbb E\Big[\sum_{t=1}^T\ell_{t,I_t}\Big]-\min_{i\in[d]}\sum_{t=1}^T\ell_{t,i}\ }\qquad
\text{and}\qquad \mathbb E[\ell_{t,I_t}]=\mathbb E\,\langle p_t,\ell_t\rangle .
$$

* Same notion as before: compare with the best **fixed** arm in hindsight.
* **What makes it hard:** the benchmark arm may be one you rarely or never played; you must be competitive with losses you **never observed**. Randomness plus the fact that the adversary committed in advance is what makes this possible.

---

## 7. The central idea: estimate the loss vector (importance weighting)

### 7.1 Reduction to Hedge via a manufactured loss vector
If we had $\ell_t$ we would run **Hedge**. We only have the single number $\ell_{t,I_t}$ and the probability $p_{t,I_t}$ with which $I_t$ was drawn. Construct an **estimated** loss vector $\hat\ell_t\in\mathbb R^d$ and feed *that* to Hedge. (The lecture first calls it a "fake" loss vector, then prefers "estimate of the ground truth".)

### 7.2 The estimator
$$
\boxed{\ \hat\ell_{t,i}=\begin{cases}\dfrac{\ell_{t,i}}{p_{t,i}}&\text{if }i=I_t\\[2mm]0&\text{if }i\ne I_t\end{cases}\ \ =\ \frac{\ell_{t,i}}{p_{t,i}}\,\mathbf 1[I_t=i]\ }
$$

So $\hat\ell_t=\big(0,\dots,0,\ \ell_{t,I_t}/p_{t,I_t},\ 0,\dots,0\big)$: zeros except at the played arm, whose observed loss is **divided by the probability of playing it**. This is **importance-weighted (importance-reweighted) sampling**, an old idea in statistics.

### 7.3 Unbiasedness
Fix $p_t$ (condition on the past). Coordinate $i$ of $\hat\ell_t$ equals $\ell_{t,i}/p_{t,i}$ with probability $p_{t,i}$ and $0$ otherwise:

$$
\mathbb E\big[\hat\ell_{t,i}\mid p_t\big]=p_{t,i}\cdot\frac{\ell_{t,i}}{p_{t,i}}+(1-p_{t,i})\cdot0=\ell_{t,i}
\ \ \Longrightarrow\ \ \boxed{\ \mathbb E[\hat\ell_t\mid p_t]=\ell_t\ }
$$

$\hat\ell_t$ is a **conditionally unbiased estimator** of the true loss vector (conditioned on $p_t$, which depends on the random past).

**Interpretation.** Freeze round $t$ and repeat it infinitely often with the same $p_t$: the *average* of the estimated vectors is exactly $\ell_t$. In the real run only **one** of the possible estimates occurs, but its average behaviour matches full information.

### 7.4 Worked example (3 arms)
Take $p_t=(0.3,\,0.3,\,0.4)$ and true losses $\ell_t=(0.1,\,0.5,\,0.8)$. Only one arm is drawn:

| Arm drawn (prob) | Observed loss | $\hat\ell_t$ |
|---|---|---|
| 1 ($0.3$) | $0.1$ | $(0.1/0.3,\,0,\,0)=(0.333,0,0)$ |
| 2 ($0.3$) | $0.5$ | $(0,\,0.5/0.3,\,0)=(0,1.667,0)$ |
| 3 ($0.4$) | $0.8$ | $(0,\,0,\,0.8/0.4)=(0,0,2)$ |

Average: $0.3(0.333,0,0)+0.3(0,1.667,0)+0.4(0,0,2)=(0.1,\,0.5,\,0.8)=\ell_t$ ✔.

### 7.5 Caveat: the estimate is not in $[0,1]^d$
$\hat\ell_{t,I_t}=\ell_{t,I_t}/p_{t,I_t}$ can be huge when $p_{t,I_t}$ is small (a rare arm happens to be drawn: if $p=10^{-4}$ the loss is multiplied by $10^4$). Hedge's analysis assumed bounded losses, so it is unclear that the algorithm works. §9 resolves this.

---

## 8. The algorithm (EXP3)

Initialise $w_{1,i}=1$ for all $i$. For $t=1,\dots,T$:
1. $p_t=\dfrac{w_t}{\lVert w_t\rVert_1}$ (normalised weights; $\lVert w_t\rVert_1=\sum_iw_{t,i}$).
2. Draw $I_t\sim p_t$ and observe $\ell_{t,I_t}\in[0,1]$.
3. Build $\hat\ell_t$ as in §7.2.
4. **Hedge update:**
   
$$
w_{t+1,i} = w_{t,i} e^{-\eta \hat{\ell}_{t,i}}
\quad \text{for all } i.
$$

Equivalently, since $\hat\ell_{t,i}=0$ for $i\ne I_t$:

$$
w_{t+1,i} = w_{t,i}
\quad (i \neq I_t)
$$

$$
w_{t+1,I_t} =
w_{t,I_t}
\exp\left(
-\eta \frac{\ell_{t,I_t}}{p_{t,I_t}}
\right)
$$

(Losses in $[0,1]$ here; any bound $[0,B]$ only rescales constants.)

It *is* Hedge, with the true loss vector replaced by an unbiased estimate; it is *not* Hedge because the estimate is random.

---

## 9. Regret analysis of EXP3

### 9.1 Where the Hedge proof used boundedness
Recall Hedge (Week 1–4 §13), valid for **any non-negative loss vectors** $\hat\ell_t$ and any $q\in\Delta_d$:

$$
\sum_{t=1}^{T}\sum_{i=1}^{d} p_{t,i}\hat{\ell}_{t,i}-
\sum_{t=1}^{T}\sum_{i=1}^{d} q_i\hat{\ell}_{t,i}
\leq
\frac{\ln(d)}{\eta}+
\frac{\eta}{2}
\sum_{t=1}^{T}\sum_{i=1}^{d}
p_{t,i}(\hat{\ell}_{t,i})^2
\qquad (A)
$$

(The proof only used $e^{-x}\le1-x+\frac{x^2}2$ for $x\ge0$, true for any non-negative loss, however large.)

**Hedge then finished by throwing $p_{t,i}$ away:** $\sum_ip_{t,i}\hat\ell_{t,i}^2\le\sum_i\ell_{t,i}^2\le d$ using $p_{t,i}\le1$ and $\ell\le 1$.

**Here that step is blocked:** $\hat\ell_{t,i}=\ell_{t,i}/p_{t,i}$ is unbounded, so $\lVert\hat\ell_t\rVert^2\le d$ is false.

**Idea:** do **not** discard $p_{t,i}$. The $p$ inside $\hat\ell$ may cancel with the $p$ in front.

### 9.2 Take expectations
The regret is an expectation, so take $\mathbb E$ of (A) (over all randomness). The left side is the regret expression **after replacing $\hat\ell$ by $\ell$**:

* $\mathbb E\langle p_t,\hat\ell_t\rangle=\mathbb E\big[\langle p_t,\mathbb E[\hat\ell_t\mid p_t]\rangle\big]=\mathbb E\langle p_t,\ell_t\rangle$ (condition on $p_t$, use unbiasedness; the outer $\mathbb E$ is over the history that determines $p_t$).
* For a fixed $q$ (non-random), $\mathbb E\langle q,\hat\ell_t\rangle=\langle q,\ell_t\rangle$ (the loss matrix is fixed in advance, so $\ell_t$ is not random).

So

$$
\mathbb{E}\left[
\sum_{t=1}^{T}\sum_{i=1}^{d}p_{t,i}\ell_{t,i}
\right]-
\sum_{t=1}^{T}\sum_{i=1}^{d}q_i\ell_{t,i}
\leq
\frac{\ln(d)}{\eta}+
\frac{\eta}{2}
\sum_{t=1}^{T}
\mathbb{E}\left[
\sum_{i=1}^{d}p_{t,i}(\hat{\ell}_{t,i})^2
\right]
\qquad (B)
$$

The left side, with $q=e_i$ for the best arm $i$, is exactly the bandit regret $R_T$.

### 9.3 The key computation: the probabilities cancel
Condition on $p_t$. For a fixed arm $i$: $\hat\ell_{t,i}^2=\dfrac{\ell_{t,i}^2}{p_{t,i}^2}$ with probability $p_{t,i}$ (when $I_t=i$) and $0$ otherwise. So

$$
\mathbb E\big[p_{t,i}\,\hat\ell_{t,i}^{\,2}\ \big|\ p_t\big]
=p_{t,i}\cdot\Big(p_{t,i}\cdot\frac{\ell_{t,i}^2}{p_{t,i}^2}\Big)
=\ell_{t,i}^2 .
$$

Summing over $i$:

$$
\boxed{\ \mathbb E\Big[\sum_ip_{t,i}\hat\ell_{t,i}^{\,2}\ \Big|\ p_t\Big]=\sum_{i=1}^d\ell_{t,i}^2\ \le\ d\ }
$$

(since each $\ell_{t,i}\in[0,1]$). The unbounded $\hat\ell$ is tamed *in expectation* because the $p$ in the denominator is cancelled by the $p$'s that multiply it (one from the weight $p_{t,i}$ in the inner product, one from the probability of drawing $i$).

### 9.4 Plug in and tune $\eta$

$$
\boxed{\ R_T\le\frac{\ln d}{\eta}+\frac{\eta}{2}\,T\,d\ }
$$

Minimise over $\eta$:

$$
-\frac{\ln d}{\eta^2}+\frac{Td}{2}=0\ \Rightarrow\ \eta^*=\sqrt{\frac{2\ln d}{Td}} .
$$

Substitute back (both terms equal $\frac12\sqrt{2Td\ln d}$):

$$
\boxed{\ R_T(\text{EXP3})\le\sqrt{2\,T\,d\,\ln d}=O\big(\sqrt{T\,d\,\ln d}\big)\ }
$$

**Reading the result:** sublinear in $T$ (regret $T\to0$), so EXP3 is a **no-regret algorithm for adversarial bandits**. Compared with Hedge ($\sqrt{2T\ln d}$, for $\ell\in[0,1]$) it pays an extra factor $\mathbf{\sqrt d}$. (The lecture says "extra $d$ factor" loosely: it is $d$ inside the square root.)

**Lower bound.**

The lecture says one can show that $\sqrt{T}$ cannot be improved (no proof given).

*(Beyond the lecture: the minimax lower bound is*

$$
\Omega(\sqrt{Td})
$$

*so EXP3 is optimal up to a factor of $\sqrt{\ln d}$.)*

---

## 10. Why "EXP3"? Exploration and exploitation

**Name:** **Exp**onentiation (the $e^{-\eta\cdot}$ update), **Exp**loration, **Exp**loitation; "3" because three words start with "Exp". Auer, Cesa-Bianchi, Freund, Schapire (2002); one of the first algorithms with sublinear regret for adversarial bandits.

### 10.1 The explore–exploit trade-off
* **Exploit:** keep playing arms that look good so far.
* **Explore:** try other arms, because with partial information you don't know their fate and the benchmark is the best arm in hindsight.
* Too much exploiting: you may miss that another arm is better. Too much exploring: you waste rounds on bad arms.

EXP3 has no explicit "explore" step; the trade-off is built into the update.

### 10.2 Where it happens
Look at the update:

* Arms **not** played: $\hat\ell_{t,i}=0\Rightarrow w_{t+1,i}=w_{t,i}$ (**unchanged**).
* Arm **played** ($I_t$): $w_{t+1,I_t}=w_{t,I_t}\,e^{-\eta\ell_{t,I_t}/p_{t,I_t}}\le w_{t,I_t}$ (**shrinks**, strictly unless $\ell_{t,I_t}=0$).

After renormalising to get $p_{t+1}$: the played arm's probability **drops** and every other arm's probability **rises** (verified in the code). So *exploiting* an arm implicitly *pushes probability towards the others* (exploration). It does not force a different arm next round, it only raises their chances.

**How much is given away** depends on the factor $\eta\,\ell_{t,I_t}/p_{t,I_t}$:
* If the arm has **zero loss**: nothing changes (no probability given away).
* If the arm had **high probability and small loss**: little is given away (a trusted arm that lives up to its reputation keeps its mass: exploitation).
* If a **rare** arm ($p$ small) is drawn with positive loss: the multiplier $1/p$ makes the penalty large (a rare event is not trusted, you push it down hard).

---

## 11. The learning rate: bandit vs full information

| | Hedge (full info) | EXP3 (bandit) |
|---|---|---|
| Information per round | all of $\ell_t$ | one coordinate |
| Best $\eta$ | $\sqrt{2\ln d/T}$ (for $\ell\in[0,1]$) | $\sqrt{2\ln d/(Td)}$ |
| Regret | $\sqrt{2T\ln d}$ | $\sqrt{2Td\ln d}$ |
| Ratio $\eta_{\text{Hedge}}/\eta_{\text{EXP3}}$ | $\sqrt d$ | |

* $\eta$ controls how aggressively weights are changed after a loss: EXP3 learns **$\sqrt d$ times slower**.
* **Intuition:** $\hat\ell_t$ is only a noisy **estimate** of $\ell_t$; with partial information you cannot trust it as much as the truth, so you update more cautiously. Using Hedge's rate with $\hat\ell$ would still give a bound, but a worse one.

---

## 12. Expected regret vs high-probability regret

* What EXP3 gives is **expected regret**: averaged over many runs (over the algorithm's randomness), regret $\le\sqrt{2Td\ln d}$.
* It does **not** say that most runs are within that bound; the regret could have **large variance**.
* A **high-probability bound** has the form
$$
\Pr\big[R_T\le\epsilon\big]\ge1-\delta\qquad(\delta\text{ small, e.g. }0.01\Rightarrow\text{ confidence }99\%).
$$
* **Why EXP3 can't give one:** the importance-weighted estimates have large variance (the $1/p$ blow-up), so the tails of the regret are hard to control; the expectation trick sidestepped this, but a tail bound needs the variance.

---

## 13. EXP3-IX (implicit exploration) and EXP3.P (explicit exploration)

### 13.1 EXP3-IX (Neu, NeurIPS 2015)
**Only change:** add $\gamma>0$ to the denominator of the estimate:

$$
\boxed{\ \hat\ell_{t,i}=\frac{\ell_{t,i}}{p_{t,i}+\gamma}\,\mathbf 1[I_t=i]\ }
$$

* The denominator is at least $\gamma$, so $\hat\ell_{t,i}\le\frac1\gamma$: **bounded**, which controls the variance and enables **martingale-based** high-probability analysis.
* The estimate is now slightly **biased downwards** (optimistic), hence "implicit exploration": rarely-played arms are penalised less.
* Everything else (Hedge update, normalisation) is as in EXP3; $\gamma$ is tuned (in the analysis $\gamma\propto\eta$).
* **Guarantee (stated, not proved in the lecture):** with probability $\ge1-\delta$,
$$
R_T\ \lesssim\ 4\sqrt{T\,d\ln d}+\Big(2\sqrt{\tfrac{Td}{\ln d}}+1\Big)\ln\frac2\delta\ =\ O(\sqrt{Td\ln d})
$$
*(the transcript garbles the formula; this is the form from Neu's paper as I recall it, so verify before quoting exact constants)*. The first term is the usual EXP3 rate; the second is a **confidence penalty**, growing only like $\ln(2/\delta)$. For a fixed confidence level the order is still $\sqrt{Td\ln d}$.
* Interpretation: run the game many times (e.g. $10^6$ plays of $T$ rounds each); with $\delta=0.01$ about $99\%$ of runs have regret under the bound, with a multiplicative constant of order $\ln 200$.

### 13.2 EXP3.P (explicit exploration; earlier line of work)
Mix the EXP3 distribution with the uniform distribution:
$$
p_{t,i}=(1-\gamma)\frac{w_{t,i}}{\sum_jw_{t,j}}+\frac\gamma d .
$$
* With prob. $\gamma$ you **explore uniformly at random**; every arm has $p_{t,i}\ge\gamma/d$, so $1/p_{t,i}\le d/\gamma$.
* The loss estimate must also be modified (biased upward slightly); the lecture skips the details.
* Also gives a high-probability bound. **EXP3-IX is preferred** because it needs only one tiny change.

| | EXP3 | EXP3-IX | EXP3.P |
|---|---|---|---|
| Distribution | $w/\sum w$ | $w/\sum w$ | $(1-\gamma)w/\sum w+\gamma/d$ |
| Estimate | $\ell/p$ | $\ell/(p+\gamma)$ | modified (biased) |
| Guarantee | expected regret $O(\sqrt{Td\ln d})$ | high probability $O(\sqrt{Td\ln d})$ | high probability |

---

## 14. Where next

* **Contextual multi-armed bandits** (adversarial setting), next lecture: the arm choice also depends on side information.
* **Stochastic bandits** later in Part 2.

---

## 15. Formula table

| # | Concept | Formula | Notes |
|---|---|---|---|
| 1 | Bandit feedback | observe only $\ell_{t,I_t}\in[0,1]$ | vs full vector in Hedge |
| 2 | Sampling | $I_t\sim p_t$, $p_t=w_t/\lVert w_t\rVert_1$ | $p_t$ is itself random |
| 3 | Bandit regret | $R_T=\mathbb E\sum_t\ell_{t,I_t}-\min_i\sum_t\ell_{t,i}$ | oblivious adversary |
| 4 | Expected loss | $\mathbb E[\ell_{t,I_t}\mid p_t]=\langle p_t,\ell_t\rangle$ | |
| 5 | IW estimate | $\hat\ell_{t,i}=\frac{\ell_{t,i}}{p_{t,i}}\mathbf 1[I_t=i]$ | zeros except played arm |
| 6 | Unbiasedness | $\mathbb E[\hat\ell_t\mid p_t]=\ell_t$ | $p\cdot\frac\ell p+(1-p)\cdot0$ |
| 7 | EXP3 update | $w_{t+1,i}=w_{t,i}e^{-\eta\hat\ell_{t,i}}$ | only played arm changes |
| 8 | Hedge inequality | $\sum\langle p_t,\hat\ell_t\rangle-\sum\langle q,\hat\ell_t\rangle\le\frac{\ln d}\eta+\frac\eta2\sum\langle p_t,\hat\ell_t^2\rangle$ | any non-negative $\hat\ell$ |
| 9 | Key cancellation | $\mathbb E[\sum_ip_{t,i}\hat\ell_{t,i}^2\mid p_t]=\sum_i\ell_{t,i}^2\le d$ | $p\cdot p\cdot\frac{\ell^2}{p^2}$ |
| 10 | EXP3 regret | $R_T\le\frac{\ln d}\eta+\frac{\eta Td}{2}$ | |
| 11 | Optimal $\eta$ | $\eta^*=\sqrt{\frac{2\ln d}{Td}}$ | $=\eta^*_{\text{Hedge}}/\sqrt d$ |
| 12 | Tuned regret | $R_T\le\sqrt{2Td\ln d}$ | vs Hedge $\sqrt{2T\ln d}$ |
| 13 | Played-arm update | $w_{t+1,I_t}=w_{t,I_t}\exp(-\eta\ell_{t,I_t}/p_{t,I_t})$ | shrinks unless loss $=0$ |
| 14 | High-prob. regret | $\Pr[R_T\le\epsilon]\ge1-\delta$ | EXP3 doesn't have it |
| 15 | EXP3-IX estimate | $\hat\ell_{t,i}=\frac{\ell_{t,i}}{p_{t,i}+\gamma}\mathbf 1[I_t=i]\le\frac1\gamma$ | implicit exploration |
| 16 | EXP3-IX bound (recalled) | $R_T\le4\sqrt{Td\ln d}+(2\sqrt{Td/\ln d}+1)\ln\frac2\delta$ w.p. $\ge1-\delta$ | check Neu (2015) |
| 17 | EXP3.P distribution | $p_{t,i}=(1-\gamma)\frac{w_{t,i}}{\sum_jw_{t,j}}+\frac\gamma d\ \ge\frac\gamma d$ | explicit exploration |

---

## 16. Quiz preparation tips

### 16.1 Definitions to state precisely
* Multi-armed bandit protocol (what is observed, what is hidden).
* Adversarial vs stochastic bandits; deterministic **oblivious** adversary.
* Bandit regret (expected loss minus best fixed arm).
* Importance-weighted estimator; **unbiased**; **conditionally** unbiased given $p_t$.
* Exploration vs exploitation; EXP3, EXP3-IX, EXP3.P.
* Expected regret vs high-probability regret.

### 16.2 Derivations likely to be asked
1. **Unbiasedness** of $\hat\ell_t$ (two-line computation).
2. **The $\mathbb E\langle p_t,\hat\ell_t^2\rangle=\sum_i\ell_{t,i}^2\le d$ computation.**
3. **Regret bound** $\frac{\ln d}\eta+\frac{\eta Td}2\to\eta^*\to\sqrt{2Td\ln d}$.
4. **Why Hedge's bounding step fails** and how the cancellation fixes it.
5. **Where explore/exploit appears** in the update (played arm shrinks, others unchanged, then normalise).

### 16.3 Common traps
* **Divide by $p_{t,I_t}$, not multiply.** Dividing by a small probability inflates the loss of rare arms.
* **$\hat\ell_t$ has one non-zero entry** (the played arm).
* **$\sqrt d$ vs $d$:** EXP3's regret has $\sqrt{d}$ extra *relative to Hedge* (total $\sqrt{Td\ln d}$); $\eta^*_{\text{EXP3}}=\eta^*_{\text{Hedge}}/\sqrt d$.
* **Conditioning:** unbiasedness holds **given $p_t$**; $p_t$ itself is random (depends on $I_1..I_{t-1}$). The final statement is an **expectation over all randomness**.
* **Oblivious** matters: the loss matrix is fixed in advance, so $\ell_t$ is not random and the comparator $q$ is fixed.
* **Regret vs unobserved arms:** the best arm in hindsight need not have been played much.
* **EXP3 gives expected regret only.** Don't claim a high-probability bound; that needs EXP3-IX / EXP3.P.
* **IX shifts the denominator** ($p+\gamma$), **EXP3.P mixes the distribution** ($(1-\gamma)p+\gamma/d$): don't confuse them.
* A larger learning rate is *more aggressive*; EXP3 must be *less* aggressive than Hedge.
* A deterministic algorithm cannot beat the oblivious adversary: randomisation is mandatory.

### 16.4 Quick numeric practice patterns
* Given $p_t$ and $\ell_t$, list the three possible $\hat\ell_t$'s and verify the average equals $\ell_t$ (§7.4).
* Given $d=10$, $T=5000$: $\eta^*=\sqrt{2\ln10/(5000\cdot10)}\approx0.0096$, bound $\sqrt{2\cdot5000\cdot10\ln10}\approx480$ (Hedge: $\approx152$ with $\eta\approx0.030$).
* Given a weight vector and a played arm with loss: apply the update only to that arm, renormalise, check its probability dropped and others rose.
* Max value of $\hat\ell$ in EXP3-IX: $\le1/\gamma$.

---

## 17. Self-test questions with answers

**Q1.** What is the one difference between the bandit protocol and online learning (full information)?
**A.** Only the loss $\ell_{t,I_t}$ of the played arm is observed, not the whole vector $\ell_t$.

**Q2.** Why must a bandit algorithm be randomised?
**A.** A deterministic algorithm can be simulated by the oblivious adversary (Cover), forcing linear regret.

**Q3.** Define the estimator and show it is unbiased.
**A.** $\hat\ell_{t,i}=\frac{\ell_{t,i}}{p_{t,i}}\mathbf 1[I_t=i]$; $\mathbb E[\hat\ell_{t,i}\mid p_t]=p_{t,i}\frac{\ell_{t,i}}{p_{t,i}}=\ell_{t,i}$.

**Q4.** Why can't we reuse Hedge's bound $\lVert\hat\ell_t\rVert^2\le d$?
**A.** $\hat\ell_{t,i}=\ell_{t,i}/p_{t,i}$ is unbounded when $p_{t,i}$ is small.

**Q5.** Show $\mathbb E[\sum_ip_{t,i}\hat\ell_{t,i}^2\mid p_t]\le d$.
**A.** For each $i$: $p_{t,i}\cdot\Pr[I_t=i]\cdot\frac{\ell_{t,i}^2}{p_{t,i}^2}=\ell_{t,i}^2\le1$; sum over $i$.

**Q6.** State EXP3's regret bound and optimal $\eta$.
**A.** $R_T\le\frac{\ln d}\eta+\frac{\eta Td}2$; $\eta^*=\sqrt{\frac{2\ln d}{Td}}$; $R_T\le\sqrt{2Td\ln d}$.

**Q7.** How does EXP3 compare to Hedge?
**A.** Same $\sqrt T$ rate; extra $\sqrt d$ factor in regret, learning rate $\sqrt d$ times smaller (estimate less trustworthy).

**Q8.** What happens to the weights after round $t$ in EXP3?
**A.** Unplayed arms' weights are unchanged; the played arm's weight is multiplied by $e^{-\eta\ell_{t,I_t}/p_{t,I_t}}\le1$. After normalisation the played arm's probability falls, the others' rise.

**Q9.** Explain the explore–exploit behaviour of EXP3.
**A.** Exploiting a (probable) arm and observing positive loss shifts probability mass to the other arms (implicit exploration). A high-probability arm with small loss keeps most of its mass; a rare arm drawn with positive loss is penalised heavily.

**Q10.** Why is EXP3's guarantee only in expectation, and why does it matter?
**A.** The importance-weighted estimates have high variance, so tail/high-probability control is hard; single runs may have regret far from the mean.

**Q11.** What does EXP3-IX change, and what does it buy?
**A.** Denominator $p_{t,i}+\gamma$; estimates bounded by $1/\gamma$; allows a martingale analysis giving high-probability regret $O(\sqrt{Td\ln d})$.

**Q12.** Contrast EXP3-IX with EXP3.P.
**A.** IX modifies the loss estimate (implicit exploration, tiny change). EXP3.P mixes uniform exploration into $p_t$ (so $p_{t,i}\ge\gamma/d$) and also modifies the estimate.

**Q13.** Difference between adversarial and stochastic bandits?
**A.** Adversarial: oblivious loss matrix chosen to hurt you. Stochastic: arm $i$'s feedback is i.i.d. with unknown mean $\theta_i$; the goal is to identify/play the best arm.

---

## 18. Lecture slips / things to double-check

1. **Example probabilities garbled.** The transcript reads "$.1\ 3\ 4$" for $p_t$, but the sampling probabilities used afterwards are $0.3,0.3,0.4$ (they sum to 1). I used $p_t=(0.3,0.3,0.4)$ with $\ell_t=(0.1,0.5,0.8)$; the unbiasedness result is independent of the particular numbers.
2. **"Extra $d$ factor":** the lecture says EXP3's regret is worse than Hedge's by an extra "$d$ factor". The bounds are $\sqrt{2Td\ln d}$ vs $\sqrt{2T\ln d}$: an extra **$\sqrt d$**.
3. **Notation:** "ea" $=\eta$; "lawn d" $=\ln d$; the transcript uses $\lVert w_t\rVert_1$ for $\sum_iw_{t,i}$ and sometimes $\langle p_t,\hat\ell_t^2\rangle$ with an element-wise square.
4. **Who/when:** the transcript garbles the authors of EXP3 ("R in 2002"). EXP3 is due to Auer, Cesa-Bianchi, Freund and Schapire (2002); EXP3-IX to Neu (NeurIPS 2015). *(From the literature, not the transcript.)*
5. **EXP3-IX bound:** the displayed formula in the transcript is unreadable. §13.1 gives the form I recall from Neu (2015); the lecture only claims the structure "(usual $\sqrt{Td\ln d}$ term) + (term involving $\ln(2/\delta)$)". Don't memorise exact constants for a quiz without checking the course material.
6. **Hedge's boundedness requirement:** the lecture phrases it as "Hedge needs losses in $[0,1]$", but as noted in §9.1 the inequality (A) only needs **non-negative** losses. The bounded-loss assumption enters only in the last step ($\langle p,\ell^2\rangle\le d$), which is the step EXP3 replaces.
7. **Lower bound:** "you cannot do better than $\sqrt T$" is stated without proof. *(Beyond the lecture: $\Omega(\sqrt{Td})$ is the known minimax rate.)*
8. **Variance claim:** the lecture says the variance "could be very high" to explain why no high-probability bound is proved for EXP3; this is a heuristic reason, not a proved statement.
9. **Cross-reference:** the lecture refers to Hedge's analysis as "lecture 16/17" of the course; in my notes it is §13 of the Week 1–4 notes.

---

## 19. Runnable verification code

Checks numerically: unbiasedness of the importance-weighted estimate (exact enumeration of the lecture's 3-arm example), the identity $\mathbb E\langle p,\hat\ell^2\rangle=\sum\ell^2\le d$, the explore/exploit probability shift, EXP3's expected regret versus $\sqrt{2Td\ln d}$ (against a fixed oblivious loss matrix) with Hedge for comparison and the $\sqrt d$ learning-rate ratio, EXP3-IX's bounded estimates and regret tail, and the EXP3.P probability floor. Requires `numpy`. Save as `verify_w7.py` and run `python verify_w7.py`.

