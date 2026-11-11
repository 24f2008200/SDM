# Sequential Decision Making — Weeks 1–4 Notes
**Online Learning: Experts, Regret, Hedge, and Combinatorial Action Spaces**

> Built from the lecture transcript (L2-23) and the handwritten slide on choosing $\eta$ for Hedge.
> Math is written in GitHub-flavoured LaTeX (`$...$` inline, `$$...$$` display).
> Notation: $d$ = number of experts/actions, $T$ = number of rounds, $\ln$ = natural log, $\log_2$ = base-2 log.

---

## Table of Contents
1. [One-page summary](#1-one-page-summary)
2. [Big picture: feedback paradigms](#2-big-picture-feedback-paradigms)
3. [Learning from expert advice: setup](#3-learning-from-expert-advice-setup)
4. [Realizable case: the stock-market guru](#4-realizable-case-the-stock-market-guru)
5. [No perfect expert: regret](#5-no-perfect-expert-regret)
6. [Weighted Majority (WM) and its analysis](#6-weighted-majority-wm-and-its-analysis)
7. [Cover's impossibility result and the oblivious adversary](#7-covers-impossibility-result-and-the-oblivious-adversary)
8. [Expected regret and randomization](#8-expected-regret-and-randomization)
9. [Randomized Weighted Majority (RWM)](#9-randomized-weighted-majority-rwm)
10. [Finite vs infinite horizon: the doubling trick](#10-finite-vs-infinite-horizon-the-doubling-trick)
11. [Lower bound: RWM is optimal](#11-lower-bound-rwm-is-optimal)
12. [Linear game on the simplex](#12-linear-game-on-the-simplex)
13. [Hedge algorithm and its regret bound (with the slide)](#13-hedge-algorithm-and-its-regret-bound)
14. [Combinatorial action spaces: the assignment problem](#14-combinatorial-action-spaces-the-assignment-problem)
15. [Formula table](#15-formula-table)
16. [Quiz preparation tips](#16-quiz-preparation-tips)
17. [Self-test questions with answers](#17-self-test-questions-with-answers)
18. [Lecture slips / things to double-check](#18-lecture-slips--things-to-double-check)
19. [Runnable verification code](#19-runnable-verification-code)

---

## 1. One-page summary

**Storyline of the four weeks** (each step fixes a problem in the previous one):

| Step | Setting | Goal | Algorithm | Guarantee |
|---|---|---|---|---|
| 1 | $d$ experts, one is **perfect** (guru) | minimise **mistakes** | Majority (keep consistent experts, follow majority) | $\le \lceil \log_2 d\rceil$ mistakes; **optimal** (adversary forces $\log_2 d$) |
| 2 | **No** perfect expert | Mistakes can be forced to $T$, so use **regret** vs best expert | Weighted Majority (deterministic) | $M \le 2(1+\epsilon)m^* + \frac{2\ln d}{\epsilon}$, **not** sublinear regret |
| 3 | Same, adversary sees the algorithm's output | — | — | **Cover:** no *deterministic* algorithm has sublinear regret |
| 4 | **Oblivious** adversary + **randomised** algorithm, measure **expected** regret | sublinear regret | Randomized Weighted Majority (RWM) | $\text{Regret}\le \frac{\ln d}{\epsilon}+\epsilon m^*$; with $\epsilon=\sqrt{\ln d/T}$: $\le 2\sqrt{T\ln d}$ |
| 5 | Unknown $T$ | infinite-horizon bound | **Doubling trick** (blocks $1,2,4,8,\dots$, reset weights) | $O(\sqrt{T\ln d})$ |
| 6 | Any algorithm | lower bound | Random-coin adversary, 2 dumb experts | $\Omega(\sqrt T)$ $\Rightarrow$ RWM is optimal in $T$ |
| 7 | General **linear game on simplex**, losses $\ell_t\in[0,B]^d$ | regret vs best fixed point | **Hedge**: $w_{t+1,i}=w_{t,i}e^{-\eta \ell_{t,i}}$ | $\le \frac{\ln d}{\eta}+\frac{\eta T B^2}{2}$; $\eta^*=\frac1B\sqrt{\frac{2\ln d}{T}}$ gives $\sqrt{2TB^2\ln d}$ |
| 8 | **Combinatorial** actions ($d=n!$) | efficient Hedge | Represent as permutation matrices; maintain a **doubly stochastic** matrix; **Sinkhorn** (projection) + **matching-based decomposition** (Birkhoff/Carathéodory) | still $O(\sqrt{T\log d})$ with $d=n!$, poly-time per round |

**Five ideas to remember**
1. **Regret** replaces "number of mistakes" because no algorithm can bound mistakes against an adversary.
2. **Randomisation** is what makes sublinear regret possible: it *expands the algorithm's decision space* from $\{0,1\}$ to $[0,1]$ (the probability $p_t$), while the adversary stays in $\{0,1\}$.
3. **Potential-function proof template**: track $\Phi_t=\sum_i w_{t,i}$; upper-bound it using the algorithm's loss; lower-bound it by the best expert's weight; take logs; use $e^x$/$\ln(1-x)$ inequalities.
4. **Optimal tuning** of the learning rate balances $\frac{\ln d}{\eta}$ (decreases in $\eta$) against $\eta T B^2/2$ (increases in $\eta$), giving $\sqrt T$ regret.
5. **Hedge can be efficient for exponentially many actions** if you can (a) project onto the action polytope and (b) decompose a point into few extreme points.

---

## 2. Big picture: feedback paradigms

| Paradigm | Feedback | Example |
|---|---|---|
| **Online learning** (full information) | You learn the outcome/loss of **every** action | Stock market: you see whether the market went up, so you know every expert's correctness |
| **Multi-armed bandits** (partial information) | You only see the loss of the action you **took** | Choosing one ad and seeing only its click |
| **Reinforcement learning** (partial information **+ state change**) | Partial feedback and your actions change the future state | Game playing, robotics |

The next several weeks focus on **online learning**.

---

## 3. Learning from expert advice: setup

* There are $d$ experts, indexed $1,\dots,d$. Single stock; each morning every expert says **invest ($1$)** or **don't invest ($0$)**.
* In round $t=1,\dots,T$:
  1. Receive advice $x_t\in\{0,1\}^d$ ($x_{t,k}$ = advice of expert $k$).
  2. Algorithm predicts $\hat y_t\in\{0,1\}$.
  3. Truth $y_t\in\{0,1\}$ is revealed (market went up = 1, down = 0).
  4. **Mistake** iff $\hat y_t\ne y_t$ (both "invested but market fell" and "didn't invest but market rose" cost 1).
* **Full information**: you learn $y_t$, hence you learn which experts were right, not just whether *you* were right.
* The algorithm has **no other information** besides expert advice.

---

## 4. Realizable case: the stock-market guru

**Assumption.** One (unknown) expert is a *guru* who is **always** correct. Goal: **minimise the number of mistakes**.

**Key observation.** Finding the guru quickly $\Leftrightarrow$ making few mistakes (after the guru is known, no more mistakes).

### 4.1 Majority algorithm
Maintain a set $C_t$ of experts still consistent with all past truths ($C_1=\{1,\dots,d\}$).

* Predict $\hat y_t=$ **majority vote** of $C_t$ (ties broken arbitrarily).
* After seeing $y_t$: set $C_{t+1}=\{k\in C_t: x_{t,k}=y_t\}$ (**eliminate** every expert who was wrong).

### 4.2 Upper bound
When the algorithm makes a mistake, the majority of $C_t$ was wrong, so **at least half** of $C_t$ is eliminated: $|C_{t+1}|\le |C_t|/2$. The guru is never eliminated, so $|C_t|\ge 1$. Hence

$$
\boxed{\ \text{Mistakes}(\text{Majority})\le \lceil \log_2 d\rceil\ }
$$

(Independent of $T$. Example: $d=10^7$ experts $\Rightarrow$ about $24$ mistakes.)

### 4.3 Lower bound (adversary argument) — Majority is optimal
Wear the **adversary hat**: control both the experts and the market, but must stay consistent with *some* guru.

* Round 1: make $d/2$ experts say 1 and $d/2$ say 0. Whatever the algorithm predicts, declare it wrong. The guru must be among the $d/2$ experts who said the other thing.
* Repeat on the surviving $d/2$ experts, and so on.
* After $\log_2 d$ rounds one expert remains. The algorithm has made $\ge\log_2 d$ mistakes **no matter what it is**.

So $\log_2 d$ is a **lower bound for every algorithm** and Majority matches it $\Rightarrow$ **Majority is optimal** in the realizable setting.

### 4.4 Why this is unrealistic
There is no infallible stock expert (the market is too complex, and a true guru would not share advice). Majority is fragile: one mistake by the (real) best expert eliminates it forever.

---

## 5. No perfect expert: regret

### 5.1 Mistakes are the wrong goal
If **no restriction** is placed on the adversary (it controls experts and market), it can declare the algorithm wrong in **every** round, so any algorithm makes $T$ mistakes. Minimising absolute mistakes is **impossible**.

### 5.2 Compare with the experts (relative performance)
You cannot blame the algorithm if all experts are bad, but you *can* if some expert is good and the algorithm fails to exploit it. So compare to the **best expert in hindsight**:

$$
\boxed{\ R_T(\mathcal A)=\underbrace{\sum_{t=1}^{T}\mathbf 1[\hat y_t\ne y_t]}_{\text{algorithm's mistakes}}-\underbrace{\min_{k\in[d]}\sum_{t=1}^{T}\mathbf 1[x_{t,k}\ne y_t]}_{\text{best expert's mistakes } m^*_T}\ }
$$

* Called **regret** because it is how much you "regret" not following the best expert from day 1.
* *Example:* $T=10{,}000$, algorithm makes $4000$ mistakes, best expert makes $400$: regret $=3600$.
* The best expert can **change with $T$**; the regret is always measured against the best expert *at the stopping time*.

### 5.3 The goal: no-regret (sublinear regret)

$$
\frac{R_T}{T}\longrightarrow 0\quad (T\to\infty)\qquad\Longleftrightarrow\qquad R_T=o(T).
$$

Per-round average regret vanishes: the algorithm "catches up" with the best expert.

### 5.4 Sanity check (consistency with the guru case)
If a guru exists then $m^*=0$ and $R_T=$ #mistakes $\le\log_2 d$, so $R_T/T\le \log_2 d/T\to0$. Majority is therefore a no-regret algorithm in the realizable case, and regret **generalises** the earlier definition.

---

## 6. Weighted Majority (WM) and its analysis

**Why not Majority?** Eliminating an expert after one mistake is wrong: an expert who errs now may be the best expert later. Instead **down-weight** instead of deleting.

### 6.1 Algorithm 
parameter is $\epsilon\in\{\(0,1\)\} $ 


Initialise $w_{1,k}=1$ for all $k$. For $t=1,\dots,T$:

1. Receive $x_t\in\{0,1\}^d$.
2. Predict $\hat y_t=1$ if $\sum_{k:x_{t,k}=1}w_{t,k}\ \ge\ \sum_{k:x_{t,k}=0}w_{t,k}$, else $0$ (**weighted majority**).
3. Observe $y_t$. **If** $\hat y_t\ne y_t$: for every expert with $x_{t,k}\neq y_t$ (the wrong ones), $w_{t+1,k}=(1-\epsilon)w_{t,k}$; others unchanged. If no mistake, weights unchanged.

Special case: $\epsilon=1$ gives back plain Majority (wrong experts get weight 0). $\epsilon=0$ does nothing. So $\epsilon\in(0,1)$ interpolates.

> **Note (lecture wording).** The lecture sketches the update in the round where the algorithm errs; the standard WM penalises every wrong expert in every round. The analysis below (only mistake rounds shrink $\Phi_t$) is stated for the mistake-round update. For the tuned bound the difference does not matter. The lower bound $(1-\epsilon)^{m_k}$ on a weight requires penalising expert $k$ on *its* mistakes, which is the standard version.

### 6.2 Analysis (potential function $\Phi_t=\sum_k w_{t,k}$)

**Upper bound on $\Phi$.** $\Phi_1=d$. In a round where the algorithm errs, the experts that were wrong held the **weighted majority**, i.e. at least half of $\Phi_t$. They lose a fraction $\epsilon$ of their weight:

$$
\Phi_{t+1}\le \Phi_t\Big(1-\tfrac{\epsilon}{2}\Big).
$$

After $M$ algorithm mistakes: $\ \Phi_{T+1}\le d\,(1-\epsilon/2)^M$. $\quad(\star)$

**Lower bound on $\Phi$.** $\Phi_{T+1}\ge w_{T+1,k^*}=(1-\epsilon)^{m^*}$ where $k^*$ is the best expert. $\quad(\star\star)$

**Combine and take logs:**

$$
m^*\ln(1-\epsilon)\le \ln d + M\ln\!\Big(1-\tfrac{\epsilon}{2}\Big).
$$

Use the two standard inequalities

$$
\ln(1-x)\le -x\quad(\text{from }e^{-x}\ge 1-x),\qquad \ln(1-x)\ge -x-x^2\ \ \text{for } x\in[0,\tfrac12].
$$

to get

$$
-m^*(\epsilon+\epsilon^2)\le \ln d-\tfrac{\epsilon M}{2}
\ \Longrightarrow\
\boxed{\ M\le 2(1+\epsilon)\,m^*+\frac{2\ln d}{\epsilon}\ }
$$

Equivalently $\ R_T(\text{WM})\le m^*+2\epsilon m^*+\frac{2\ln d}{\epsilon}$.

### 6.3 Why this is *not* a no-regret guarantee
The bound contains a **$m^*$ term with constant 1** (plus $2\epsilon m^*$). If even the best expert errs linearly ($m^*=\Theta(T)$), $R_T/T$ does **not** tend to $0$. So WM gives a regret bound, but **not sublinear regret**.

**Three possible explanations** the lecture considered:
1. WM is a bad algorithm. ✗
2. Our analysis was not tight enough. ✗
3. **No** (deterministic) algorithm can achieve sublinear regret. ✔ (next section)

---

## 7. Cover's impossibility result and the oblivious adversary

### 7.1 Cover's example (negative result)
Two experts: expert 1 **always says 1**, expert 2 **always says 0**. Adversary sees the algorithm's prediction and sets $y_t=1-\hat y_t$.

* Algorithm makes $T$ mistakes.
* Expert 1 errs $L$ times, expert 2 errs $T-L$ times (they err on complementary rounds). Best expert errs $\min(L,T-L)\le T/2$ times.

$$
R_T\ \ge\ T-\tfrac T2=\tfrac T2\quad\Longrightarrow\quad \frac{R_T}{T}\ge \frac12\not\to0.
$$

**No deterministic algorithm** (WM included) can achieve sublinear regret against this adversary.

### 7.2 Restricting the adversary: the oblivious adversary
Reducing the adversary all the way to "must be consistent with a guru" is too weak a model; leaving it unrestricted is too strong. Middle ground:

> **Oblivious adversary.** The algorithm's **pseudo-code is published**, and the adversary fixes the whole sequence $y_1,y_2,\dots$ **before the game starts**. It does **not** see the algorithm's **random bits** (realised coin tosses).

* If the algorithm is **deterministic**, the adversary can simulate it in advance and Cover's argument still applies. So randomness is **the key ingredient**.
* A randomised algorithm may use known probabilities; the adversary knows these but not their realisation.
* From now on only **randomised** algorithms are considered. WM is deterministic, so it is natural that it failed.

---

## 8. Expected regret and randomization

The algorithm now outputs $\hat y_t$ at random, so $R_T$ is a random variable. Use the **expected** regret:

$$
\mathbb E[R_T]=\mathbb E\Big[\sum_{t=1}^T\mathbf 1[\hat y_t\ne y_t]\Big]-\min_k\sum_{t=1}^T\mathbf 1[x_{t,k}\ne y_t].
$$

(The expectation is over the algorithm's randomness; the adversary may also be random, but you can think of it as deterministic for the analysis.)

By **linearity of expectation** and $\mathbb E[\mathbf 1[A]]=\Pr(A)$:

$$
\mathbb E\Big[\sum_t\mathbf 1[\hat y_t\ne y_t]\Big]=\sum_{t=1}^T\Pr(\hat y_t\ne y_t).
$$

### 8.1 Key identity
Let $p_t=\Pr(\hat y_t=1)$.

* If $y_t=1$: $\Pr(\hat y_t\ne y_t)=1-p_t$.
* If $y_t=0$: $\Pr(\hat y_t\ne y_t)=p_t$.

In both cases

$$
\boxed{\ \Pr(\hat y_t\ne y_t)=\lvert p_t-y_t\rvert\ }
$$

so

$$
\mathbb E[R_T]=\sum_{t=1}^T \lvert p_t-y_t\rvert-\min_k\sum_{t=1}^T\mathbf 1[x_{t,k}\neq y_t].
$$

### 8.2 Intuition: randomness expands the algorithm's decision space
The algorithm now "plays" $p_t\in[0,1]$ (a continuum), while the truth $y_t\in\{0,1\}$ stays binary. The loss $\lvert p_t-y_t\rvert$ is a *continuous* function of the decision. This is what indirectly **reduces the adversary's power**.

---

## 9. Randomized Weighted Majority (RWM)

### 9.1 Algorithm
Same weights and same update as WM; only the **prediction rule** changes:

$$
\boxed{\ p_t=\Pr(\hat y_t=1)=\frac{\sum_{k:\,x_{t,k}=1}w_{t,k}}{\sum_{k=1}^d w_{t,k}}\ },\qquad \hat y_t\sim\text{Bernoulli}(p_t).
$$

Update: $w_{t+1,k}=(1-\epsilon)w_{t,k}$ for every expert $k$ with $x_{t,k}\ne y_t$, else unchanged. Start with $w_{1,k}=1$.

**Interpretation:** pick expert $k$ with probability $w_{t,k}/\Phi_t$ and copy their advice.

### 9.2 Analysis
$\Phi_{t+1}=\Phi_t-\epsilon\sum_{k:x_{t,k}\neq y_t}w_{t,k}$. The sum of weights of wrong experts is $\Phi_t\cdot \Pr(\text{mistake at }t)=\Phi_t\lvert p_t-y_t\rvert$. So

$$
\Phi_{t+1}=\Phi_t\big(1-\epsilon\lvert p_t-y_t\rvert\big)\ \Longrightarrow\ \Phi_{T+1}=d\prod_{t=1}^T\big(1-\epsilon\lvert p_t-y_t\rvert\big).
$$

(Compare WM: the factor $(1-\epsilon/2)$ becomes $(1-\epsilon\lvert p_t-y_t\rvert)$. The "2" from "majority" disappears.)

Lower bound: $\Phi_{T+1}\ge(1-\epsilon)^{m^*}$. Take logs, use $\ln(1-x)\le -x$ on the product side and $\ln(1-x)\ge -x-x^2$ on the other:

$$
-m^*(\epsilon+\epsilon^2)\le\ln d-\epsilon\sum_{t}\lvert p_t-y_t\rvert
$$

$$
\boxed{\ \mathbb E[R_T(\text{RWM})]=\sum_t\lvert p_t-y_t\rvert-m^*\ \le\ \frac{\ln d}{\epsilon}+\epsilon\, m^*\ }
$$

Compare with WM: no extra $m^*$ term. This is the improvement.

### 9.3 Choosing $\epsilon$
Using $m^*\le T$: minimise $f(\epsilon)=\frac{\ln d}{\epsilon}+\epsilon T$.

$$
f'(\epsilon)=-\frac{\ln d}{\epsilon^2}+T=0\ \Rightarrow\ \epsilon^*=\sqrt{\frac{\ln d}{T}}
$$

$$
\boxed{\ \mathbb E[R_T(\text{RWM})]\le 2\sqrt{T\ln d}\ }\qquad\Rightarrow\qquad \frac{R_T}{T}\le 2\sqrt{\frac{\ln d}{T}}\to0 .
$$

(Valid when $\epsilon^*\le\tfrac12$, i.e. $T\ge4\ln d$, since $\ln(1-x)\ge-x-x^2$ was used on $[0,\frac12]$.)

**Caveat:** $\epsilon^*$ depends on $T$, so this is a **finite-horizon** bound: you must know $T$ in advance.

---

## 10. Finite vs infinite horizon: the doubling trick

**Goal:** a bound valid for *every* stopping time $T$, without knowing $T$.

### 10.1 Equal-size blocks fail
Split time into blocks of fixed size $K$, reset weights after each block, use $\epsilon=\sqrt{\ln d/K}$. Each block costs $\le2\sqrt{K\ln d}$; over $T/K$ blocks:

$$
R_T\le \frac TK\cdot 2\sqrt{K\ln d}=2T\sqrt{\frac{\ln d}{K}}\quad\Rightarrow\quad \frac{R_T}{T}\le 2\sqrt{\frac{\ln d}{K}}\not\to0 .
$$

Linear in $T$, so **not** no-regret.

### 10.2 Doubling trick (block sizes $1,2,4,8,\dots$)
Block $\ell$ has length $2^\ell$ ($\ell=0,1,2,\dots$); restart RWM with $\epsilon_\ell=\sqrt{\ln d/2^\ell}$ at the start of every block. If $2^k\le T<2^{k+1}$, then $T$ lies in block $k$ and

$$
R_T\le\sum_{\ell=0}^{k}2\sqrt{2^\ell\ln d}
=2\sqrt{\ln d}\;\frac{(\sqrt2)^{k+1}-1}{\sqrt2-1}
\le \frac{2\sqrt2}{\sqrt2-1}\sqrt{T\ln d}.
$$

This is a geometric sum with ratio $\sqrt 2$ dominated by the last term, and $2^{k/2}\le\sqrt T$.

$$
\boxed{\ R_T=O(\sqrt{T\ln d})\ \ \text{for all }T\ (\text{infinite horizon}),\quad \frac{R_T}{T}\to0 .}
$$

Constant $\approx 6.83$. **Doubling trick = a generic recipe** to convert a finite-horizon algorithm into an anytime algorithm at the cost of a constant factor.

---

## 11. Lower bound: RWM is optimal

Return to Cover's two experts (always 1, always 0), but now the adversary is **oblivious** and **random**: $y_t\sim\text{Bernoulli}(1/2)$ i.i.d.

* **Algorithm's expected mistakes:** whatever $p_t$ is,
 $\Pr(\hat y_t\ne y_t)=\tfrac{p_t}{2}+\tfrac{1-p_t}{2}=\tfrac12$, so $\mathbb E[\text{mistakes}]=T/2$.
* **Best expert:** expert 1 errs when $y_t=0$, expert 2 when $y_t=1$. Best expert's mistakes $=\min(N_0,N_1)$ with $N_0+N_1=T$. A (non-trivial) fact, **stated without proof** in the lecture:

$$
\mathbb E[\min(N_0,N_1)]=\frac T2-\Theta(\sqrt T).
$$

(Intuition: the number of heads in $T$ fair tosses fluctuates by about $\sqrt T$ around $T/2$, and the minimum of the two counts is below $T/2$ by that amount on average.)

Therefore, for **every** algorithm,

$$
\mathbb E[R_T]\ \ge\ \frac T2-\Big(\frac T2-\Theta(\sqrt T)\Big)=\Omega(\sqrt T).
$$

**Conclusion.** Any algorithm can be forced to suffer $\Omega(\sqrt T)$ regret; RWM achieves $O(\sqrt{T\ln d})$. Up to the dependence on $d$, **RWM is optimal**. Sublinear is achievable, but $\sqrt T$ is the best possible rate (not $\log T$).

> Beyond the lecture: the full minimax lower bound also contains the $\sqrt{\ln d}$ factor, i.e. $\Omega(\sqrt{T\ln d})$. The lecture's two-expert example only shows the $\sqrt T$ part.

---

## 12. Linear game on the simplex

### 12.1 Generalising experts
Let $\Delta_d=\{p\in\mathbb R^d:\ p_i\ge0,\ \sum_ip_i=1\}$ (probability simplex). **Protocol:** for $t=1,\dots,T$:

1. Learner plays $p_t\in\Delta_d$.
2. Adversary reveals loss vector $\ell_t\in\mathbb R^d_{+}$ (**not** restricted to $\{0,1\}$).
3. Learner suffers $\langle p_t,\ell_t\rangle=p_t^\top\ell_t$.

### 12.2 Experts problem is a special case
Take $p_t=w_t/\Phi_t$ (normalised weights) and $\ell_{t,k}=\mathbf 1[x_{t,k}\neq y_t]$ ("was expert $k$ wrong?"). Then

$$
\langle p_t,\ell_t\rangle=\sum_{k:\,x_{t,k}\ne y_t}p_{t,k}=\Pr(\hat y_t\ne y_t)=\lvert p^{\text{RWM}}_t-y_t\rvert .
$$

Worked example from the lecture: weights $(0.2,\,0.3,\,0.25)$, advice $(1,0,1)$, truth $y_t=1$.

* $\Phi_t=0.75$, $\ p_t=(0.267,\,0.4,\,0.333)$.
* $\Pr(\hat y_t=1)=\frac{0.2+0.25}{0.75}=0.6$.
* Loss vector $\ell_t=(0,1,0)$ (only expert 2 is wrong). $\langle p_t,\ell_t\rangle=0.4=1-0.6$ ✔.

The loss vector **fuses** expert advice and the truth into one vector. Hence the problem is more general: actions can be stocks, routes, search engines, etc.

### 12.3 Regret definition
$$
R_T=\sum_{t=1}^T\langle p_t,\ell_t\rangle-\min_{q\in\Delta_d}\sum_{t=1}^T\langle q,\ell_t\rangle .
$$

**The comparator can be restricted to the corners.** $\sum_t\langle q,\ell_t\rangle=\langle q,L_T\rangle$ with $L_T=\sum_t\ell_t$ (cumulative loss vector). A linear function on the simplex is minimised at a vertex (put all mass on the smallest coordinate of $L_T$):

$$
R_T=\sum_{t=1}^T\langle p_t,\ell_t\rangle-\min_{i\in[d]}\sum_{t=1}^T\ell_{t,i}.
$$

(The minimiser need not be unique, e.g. ties, but a corner always attains the minimum.)

### 12.4 Mixed play vs sampling
* If you can **play $p_t$ directly** (e.g. split ₹100 across $d$ stocks), the loss is $\langle p_t,\ell_t\rangle$.
* If you must **pick a single action** (e.g. one movie slot on Netflix), sample $X_t\sim p_t$ (a corner $e_i$). Then $\mathbb E[\langle X_t,\ell_t\rangle]=\langle p_t,\ell_t\rangle$, so the **expected** regret is identical.

---

## 13. Hedge algorithm and its regret bound

### 13.1 Algorithm (learning rate $\eta>0$)
Initialise $w_{1,i}=1$ for all $i$. For $t=1,\dots,T$:

1. $\Phi_t=\sum_iw_{t,i}$; play $p_t=w_t/\Phi_t$.
2. Receive $\ell_t$.
3. **Multiplicative update:** $\ w_{t+1,i}=w_{t,i}\,e^{-\eta\,\ell_{t,i}}$ for all $i$.

* Weights stay **strictly positive**, so $p_t$ is always in the *interior* of the simplex: Hedge **never** plays a corner exactly (while the comparator *is* a corner).
* If $\ell_{t,i}\in\{0,1\}$, Hedge $\approx$ RWM ($e^{-\eta}\approx1-\epsilon$ for small $\eta$).
* Assumption for the bound: $0\le \ell_{t,i}\le B$ for all $t,i$.
* (In the lecture $\eta$ was written "ea".)

### 13.2 Regret analysis (full derivation)
Track $\Phi_{t+1}/\Phi_t$.

**Step 1: express as an average under $p_t$.**

$$
\frac{\Phi_{t+1}}{\Phi_t}=\frac{\sum_iw_{t,i}e^{-\eta\ell_{t,i}}}{\Phi_t}=\sum_{i=1}^d p_{t,i}\,e^{-\eta\ell_{t,i}} .
$$

**Step 2: linearise the exponential.** For $x\le0$: $\ e^{x}\le1+x+\tfrac{x^2}{2}$. With $x=-\eta\ell_{t,i}\le0$:

$$
\frac{\Phi_{t+1}}{\Phi_t}\le\sum_ip_{t,i}\Big(1-\eta\ell_{t,i}+\tfrac{\eta^2\ell_{t,i}^2}{2}\Big)
=1-\eta\langle p_t,\ell_t\rangle+\tfrac{\eta^2}{2}\langle p_t,\ell_t^{2}\rangle ,
$$

where $\ell_t^2$ is the coordinate-wise square and we used $\sum_ip_{t,i}=1$.

**Step 3: use $1+x\le e^x$, then take logs.**

$$
\ln\Phi_{t+1}-\ln\Phi_t\le-\eta\langle p_t,\ell_t\rangle+\tfrac{\eta^2}{2}\langle p_t,\ell_t^2\rangle .
$$

**Step 4: sum over $t$ (telescoping).** With $\Phi_1=d$:

$$
\ln\Phi_{T+1}\le\ln d-\eta\sum_t\langle p_t,\ell_t\rangle+\tfrac{\eta^2}{2}\sum_t\langle p_t,\ell_t^2\rangle .\qquad(1)
$$

**Step 5: lower bound with the best action $i^*$.**

$$
\ln\Phi_{T+1}\ge\ln w_{T+1,i^*}=-\eta\sum_t\ell_{t,i^*}.\qquad(2)
$$

**Step 6: combine (1) and (2) and divide by $\eta$.**

$$
\underbrace{\sum_t\langle p_t,\ell_t\rangle-\sum_t\ell_{t,i^*}}_{R_T}\le\frac{\ln d}{\eta}+\frac\eta2\sum_t\langle p_t,\ell_t^{2}\rangle .
$$

**Step 7: bound the last term.** Since $\ell_{t,i}\le B$: $\langle p_t,\ell_t^2\rangle\le B^2\sum_ip_{t,i}=B^2$. Hence

$$
\boxed{\ R_T(\text{Hedge})\le\frac{\ln d}{\eta}+\frac{\eta\,T B^2}{2}\ =: C(\eta)\ }
$$

This holds for **any** $\eta>0$.

### 13.3 Choosing $\eta$ (this is the slide)
> **Slide: "Choose $\eta$ to minimize $C$."**

Differentiate $C(\eta)=\frac{\ln d}{\eta}+\frac{\eta TB^2}{2}$ and set to zero:

$$
-\frac{\ln d}{\eta^2}+\frac{TB^2}{2}=0
\ \Rightarrow\ 
\eta^2=\frac{2\ln d}{TB^2}
\ \Rightarrow\ 
\boxed{\ \eta^*=\sqrt{\frac{2\ln d}{TB^2}}=\frac1B\sqrt{\frac{2\ln d}{T}}\ }
$$

($C$ is convex in $\eta>0$, so this stationary point is the minimum.)

**Plug $\eta^*$ back into $C$:**

$$
\frac{\ln d}{\eta^*}=\ln d\cdot\sqrt{\frac{TB^2}{2\ln d}}=\frac{1}{\sqrt2}\sqrt{\ln d}\,\sqrt{TB^2},
\qquad
\frac{\eta^*TB^2}{2}=\frac12\sqrt{\frac{2\ln d}{TB^2}}\,TB^2=\frac1{\sqrt2}\sqrt{\ln d}\,\sqrt{TB^2}.
$$

The two terms are **equal** at the optimum (a general feature of this kind of tuning), so

$$
\boxed{\ \text{Regret}(\text{Hedge})\le\sqrt{2\ln(d)\,T B^2}=B\sqrt{2T\ln d}=O\!\big(\sqrt{T\ln d}\big)\ }
$$

Per round: $\frac{R_T}{T}\le B\sqrt{\frac{2\ln d}{T}}\to0$. Hedge is a **no-regret** algorithm for linear games on the simplex.

### 13.4 Remarks
* $\eta^*$ needs $T$ and $B$ in advance (finite horizon). The doubling trick fixes the $T$-dependence.
* **Trade-off:** small $\eta$ $\Rightarrow$ slow learning ($\ln d/\eta$ large). Large $\eta$ $\Rightarrow$ over-reacts to noise ($\eta TB^2/2$ large).
* Dependence on the number of actions is only $\sqrt{\ln d}$, so adding actions is almost free: that is why $d=n!$ is still fine *statistically*; the issue is *computation*.

---

## 14. Combinatorial action spaces: the assignment problem

### 14.1 Problem
$n$ vehicles, $n$ routes. Each day choose a **bijection** vehicles $\to$ routes (a **permutation**). Nature reveals an $n\times n$ matrix $L_t$ ($L_t[i,j]$ = time of vehicle $i$ on route $j$) (full information). Loss $=\sum_i L_t[i,\sigma(i)]$ for the chosen permutation $\sigma$. Regret is measured against the best **fixed** permutation in hindsight.

### 14.2 Naive Hedge breaks
There are $n!$ actions, so Hedge must keep a weight/probability for each: $p_t\in\Delta_{n!}$. For $n=100$, $100!$ is astronomically large. Statistically fine ($\sqrt{T\ln n!}\approx\sqrt{T\,n\ln n}$) but **computationally infeasible**.

*Toy example ($n=3$, 6 permutations):* matrix rows $(10,8,7.5),(18,25,3),(7,5.5,6.2)$. Identity assignment loss $=10+25+6.2=41.2$. Permutation "$V_1\to R_3,V_2\to R_1,V_3\to R_2$" has loss $7.5+18+5.5=31$. Naive Hedge builds a length-$6$ loss vector with one such sum per permutation.

### 14.3 Step A: a better representation (permutation matrices)
Represent each action by its **permutation matrix** $A\in\{0,1\}^{n\times n}$ (exactly one 1 per row and column), flattened to $\mathbb R^{n^2}$.

* **Loss stays linear:** $\text{loss}=\langle A,L_t\rangle=\sum_{i,j}A_{ij}L_t[i,j]$.
* **Dimension drops** from $n!$ to $n^2$ (for $n=100$: $10^4$).
* **But** there are still $n!$ actions, so a probability vector over them is still $n!$-dimensional. Lower-dimensional *representation* $\neq$ fewer *actions*.

### 14.4 Step B: maintain a point in the convex hull instead of a distribution
A distribution over actions corresponds to a **convex combination** $P=\sum_i\lambda_iA_i$ (with $\lambda\in\Delta_{n!}$). Expected loss only depends on $P$:

$$
\mathbb E[\langle A,L\rangle]=\Big\langle\sum_i\lambda_iA_i,\,L\Big\rangle=\langle P,L\rangle .
$$

**What is $\text{conv}\{\text{permutation matrices}\}$?**

* Any convex combination of permutation matrices is **doubly stochastic**: entries in $[0,1]$, every row sum $=1$, every column sum $=1$.
* **Birkhoff–von Neumann theorem:** the converse is true as well, so

$$
\text{conv}\{\text{permutation matrices}\}=\{\text{doubly stochastic matrices}\}=:\text{DSP (the Birkhoff polytope)}.
$$

(This is not obvious: DSP could a priori be larger than the convex hull.) So the algorithm only has to **maintain an $n\times n$ doubly stochastic matrix** $P_t$.

### 14.5 Step C: Carathéodory (few extreme points suffice)
If the algorithm must **play one permutation**, sample from a decomposition $P_t=\sum_i\lambda_iA_i$. Storing $n!$ numbers is again a problem, but:

> **Carathéodory's theorem.** A point in the convex hull of any set of points in $\mathbb R^D$ is a convex combination of **at most $D+1$** of them.

Here $D=n^2$, so $P_t$ needs at most $n^2+1$ permutation matrices. (Existence result; the decomposition is **not unique**.) Any decomposition gives the **same expected loss** $\langle P_t,L_t\rangle$, so the choice of decomposition doesn't affect regret.

*Picture: unit square in $\mathbb R^2$ with 4 corner points: any interior point needs only 3 of them (one of the two triangles).*

### 14.6 Step D: algorithmic decomposition (matching + peeling)
Given doubly stochastic $A$, repeat until $A=0$:

1. Build the **bipartite graph** with an edge $(i,j)$ iff $A_{ij}>0$.
2. Find a **perfect matching** $M$ (poly-time). It exists because $A$ (and every residual) is a *scaled* doubly stochastic matrix; matching existence follows from Birkhoff.
3. $\lambda=\min_{(i,j)\in M}A_{ij}$. Record $(\lambda,\,P_M)$ where $P_M$ is the permutation matrix of $M$.
4. $A\leftarrow A-\lambda P_M$. At least **one entry becomes 0** (more if there are ties).

*Complexity:* at most $n^2$ rounds (at least one nonzero entry is zeroed per round), each round poly-time $\Rightarrow$ **polynomial overall**. Yields $\le n^2$ permutations (best case $\approx n$ rounds if all matched entries tie).

**Worked example (verified in code):**

$$
A=\begin{pmatrix}0.2&0.3&0.5\\0.4&0.4&0.2\\0.4&0.3&0.3\end{pmatrix}
=0.2\,P_{(1,2,3)}+0.2\,P_{(2,3,1)}+0.1\,P_{(2,1,3)}+0.3\,P_{(3,1,2)}+0.2\,P_{(3,2,1)} ,
$$

where $P_{(a,b,c)}$ means vehicle $1\to$ route $a$, vehicle $2\to b$, vehicle $3\to c$. Weights sum to $0.2+0.2+0.1+0.3+0.2=1$ ✔. (Different matchings give different, equally valid decompositions.)

### 14.7 Step E: projection onto DSP (Sinkhorn balancing)
After the multiplicative update the weight matrix $W_t$ is positive but **not** doubly stochastic. We need the DSP analogue of "normalise the weight vector":

* In Hedge, $p=w/\Phi$ **is a projection onto the simplex**: "normalisation = projection". (Precisely, it is the **KL / relative-entropy projection**; the lecture says "closest in the $L_2$ sense", which is not exactly right, see §18.)
* For matrices: **Sinkhorn balancing**: *repeatedly normalise all rows to sum 1, then all columns to sum 1, and iterate*. It converges to a doubly stochastic matrix, which is the (KL) projection of $W_t$ onto DSP.
* General principle: to project onto $S_1\cap S_2$ when projecting onto each $S_i$ is easy, **alternate projections** between $S_1$ (row-normalised matrices) and $S_2$ (column-normalised matrices).

### 14.8 The combinatorial Hedge algorithm
1. Initialise $W_1=\mathbf 1\mathbf 1^\top$ (all ones, $n\times n$).
2. For $t=1,\dots,T$:
   1. **Project:** $P_t=\text{Sinkhorn}(W_t)$ (doubly stochastic).
   2. **Decompose:** $P_t=\sum_{i=1}^{\le n^2}\lambda_{t,i}M_{t,i}$ (matching-based).
   3. **Sample** permutation matrix $M_{t,i}$ with probability $\lambda_{t,i}$ and **play** it.
   4. Receive $L_t\in\mathbb R^{n\times n}$; expected loss $\langle P_t,L_t\rangle$.
   5. **Update:** $W_{t+1}[i,j]=W_t[i,j]\,e^{-\eta L_t[i,j]}$ for all $i,j$.

Same *skeleton* as Hedge: normalisation $\leftrightarrow$ projection; sampling $\leftrightarrow$ decomposition (both trivial for the simplex). Regret is still $O(\sqrt{T\ln d})$ with $d=n!$ (so $\sqrt{T\,n\ln n}$), with polynomial cost per round (the lecture states this without proof).

### 14.9 General recipe for combinatorial online learning
1. **Represent** actions as vectors such that the loss is **linear**.
2. Identify the **polytope** (convex hull of action vectors).
3. Be able to **decompose** a point of the polytope into $\le D+1$ extreme points in poly-time.
4. Be able to **project** an arbitrary positive point onto the polytope in poly-time.

Then a Hedge-like algorithm works. Other examples mentioned: **transportation polytopes** ($m\times n$ matrices), **path polytopes**.

### 14.10 Preview of the next topic
Hedge assumes **simplex actions + linear loss**. Standard ML uses other losses (e.g. squared loss $(\hat y-y)^2$ in regression), which are **not linear**. Next: online versions of batch ML problems, i.e. non-linear (convex) losses.

---

## 15. Formula table

| # | Concept | Formula | Notes |
|---|---|---|---|
| 1 | Mistake indicator | $\mathbf 1[\hat y_t\ne y_t]$ | $\hat y_t$ prediction, $y_t$ truth |
| 2 | Majority mistake bound (guru exists) | $\le\lceil\log_2 d\rceil$ | Optimal; adversary forces $\log_2d$ |
| 3 | Regret (experts) | $R_T=\sum_t\mathbf 1[\hat y_t\ne y_t]-\min_k\sum_t\mathbf 1[x_{t,k}\ne y_t]$ | vs best expert in hindsight |
| 4 | No-regret condition | $R_T/T\to0$ (sublinear, $R_T=o(T)$) | Goal |
| 5 | WM update | $w_{t+1,k}=(1-\epsilon)w_{t,k}$ for wrong experts | $\epsilon=1$: Majority |
| 6 | WM potential step | $\Phi_{t+1}\le\Phi_t(1-\epsilon/2)$ on a mistake | weighted majority was wrong |
| 7 | WM bound | $M\le2(1+\epsilon)m^*+\frac{2\ln d}{\epsilon}$ | Not sublinear |
| 8 | Cover's lower bound (deterministic) | $R_T\ge T/2$ | 2 constant experts, reactive adversary |
| 9 | Expected mistakes | $\mathbb E[\text{mistakes}]=\sum_t\Pr(\hat y_t\neq y_t)=\sum_t\lvert p_t-y_t\rvert$ | $p_t=\Pr(\hat y_t=1)$ |
| 10 | RWM prediction | $p_t=\dfrac{\sum_{k:x_{t,k}=1}w_{t,k}}{\sum_kw_{t,k}}$ | |
| 11 | RWM potential | $\Phi_{t+1}=\Phi_t\,(1-\epsilon\lvert p_t-y_t\rvert)$ | |
| 12 | RWM bound | $\mathbb E[R_T]\le\dfrac{\ln d}{\epsilon}+\epsilon m^*$ | |
| 13 | Optimal $\epsilon$ (RWM) | $\epsilon^*=\sqrt{\ln d/T}$ | needs $T$ |
| 14 | RWM tuned regret | $\mathbb E[R_T]\le2\sqrt{T\ln d}$ | |
| 15 | Doubling trick | blocks $2^0,2^1,\dots$; $R_T\le\frac{2\sqrt2}{\sqrt2-1}\sqrt{T\ln d}$ | anytime bound |
| 16 | Lower bound (any alg.) | $\mathbb E[R_T]=\Omega(\sqrt T)$ | $\mathbb E[\min(N_0,N_1)]=\frac T2-\Theta(\sqrt T)$ |
| 17 | Simplex | $\Delta_d=\{p\ge0,\ \sum_ip_i=1\}$ | |
| 18 | Linear game regret | $R_T=\sum_t\langle p_t,\ell_t\rangle-\min_i\sum_t\ell_{t,i}$ | min over simplex = min over corners |
| 19 | Hedge update | $w_{t+1,i}=w_{t,i}e^{-\eta\ell_{t,i}}$, $p_t=w_t/\Phi_t$ | |
| 20 | Key inequalities | $1+x\le e^x$; $e^x\le1+x+\tfrac{x^2}2$ ($x\le0$); $\ln(1-x)\le-x$; $\ln(1-x)\ge-x-x^2$ ($0\le x\le\tfrac12$) | used in all proofs |
| 21 | Hedge one-step | $\ln\frac{\Phi_{t+1}}{\Phi_t}\le-\eta\langle p_t,\ell_t\rangle+\frac{\eta^2}2\langle p_t,\ell_t^2\rangle$ | |
| 22 | Hedge bound | $R_T\le\dfrac{\ln d}{\eta}+\dfrac{\eta TB^2}{2}$ | $\ell_{t,i}\in[0,B]$ |
| 23 | Optimal $\eta$ (slide) | $\eta^*=\sqrt{\dfrac{2\ln d}{TB^2}}=\dfrac1B\sqrt{\dfrac{2\ln d}{T}}$ | from $C'(\eta)=0$ |
| 24 | Hedge tuned regret | $R_T\le\sqrt{2\ln(d)\,TB^2}=O(\sqrt{T\ln d})$ | both terms equal at optimum |
| 25 | Doubly stochastic | $M_{ij}\in[0,1]$, $\sum_jM_{ij}=1\ \forall i$, $\sum_iM_{ij}=1\ \forall j$ | |
| 26 | Birkhoff | $\text{conv}(\text{perm. matrices})=\text{DSP}$ | |
| 27 | Carathéodory | point in conv of points in $\mathbb R^D$ $\Rightarrow$ $\le D+1$ points | $D=n^2$: $\le n^2+1$ permutations |
| 28 | Decomposition cost | $\le n^2$ matching rounds, each poly-time | $\ge1$ entry zeroed per round |
| 29 | Combinatorial Hedge update | $W_{t+1}[i,j]=W_t[i,j]e^{-\eta L_t[i,j]}$, then Sinkhorn | |
| 30 | Combinatorial regret | $O(\sqrt{T\ln(n!)})=O(\sqrt{T\,n\ln n})$ | $\ln n!\le n\ln n$ |

---

## 16. Quiz preparation tips

### 16.1 Definitions to be able to state precisely
* **Regret** vs **mistakes**: be explicit that regret compares with the best *fixed* expert/action **in hindsight**, after $T$ rounds.
* **No-regret / sublinear regret / zero-regret algorithm:** $R_T/T\to0$.
* **Oblivious adversary:** fixes $y_1,y_2,\dots$ before the game; knows the algorithm's code, not its realised random bits.
* **Finite horizon vs infinite horizon** bound.
* **Doubly stochastic matrix**, **Birkhoff polytope**, **permutation matrix**, **bipartite graph**, **perfect matching**.

### 16.2 Proof templates (likely to be asked as "derive/fill in")
1. **Majority bound:** each mistake halves $|C_t|$ $\Rightarrow$ $\log_2d$.
2. **Potential argument** ($\Phi_t=\sum w$): upper bound by algorithm loss, lower bound by best expert $w_{i^*}$, take logs, apply the inequalities.
3. **Optimising the parameter:** differentiate, set to zero, substitute back; both terms become equal.
4. **Lower bound:** adversary argument (guru case) / random-coin adversary (regret case).

### 16.3 Common traps
* **Log base:** $\lceil\log_2d\rceil$ for Majority; $\ln d$ in WM/RWM/Hedge.
* **WM vs RWM constants:** WM has factor $2$ and an *extra* $m^*$ term. RWM has $\frac{\ln d}{\epsilon}+\epsilon m^*$ with **no** extra $m^*$.
* **$\epsilon^*$ (RWM)** $=\sqrt{\ln d/T}$ (no 2), but **$\eta^*$ (Hedge)** $=\frac1B\sqrt{2\ln d/T}$ (has a 2 and $B$). Tuned bounds: RWM $2\sqrt{T\ln d}$, Hedge $B\sqrt{2T\ln d}$.
* **Equal blocks do NOT work, doubling blocks do.** The reason: equal blocks give regret linear in $T$.
* **Determinism:** WM is deterministic $\Rightarrow$ cannot be no-regret (Cover). Randomisation is essential.
* **"Expected" regret:** $\Pr(\hat y_t\ne y_t)=\lvert p_t-y_t\rvert$, true for both $y_t=0$ and $y_t=1$.
* **Hedge never plays a corner**, but the **comparator** is a corner. Both statements are important.
* **Representation vs action count:** permutation matrices reduce *dimension* ($n!\to n^2$) but not the *number of actions* ($n!$).
* **Carathéodory is an existence result** (not unique, not algorithmic). The matching-peeling algorithm makes it constructive.
* **Birkhoff:** convex hull of permutation matrices $=$ doubly stochastic matrices (both inclusions).
* **Normalisation = projection** onto the simplex; Sinkhorn is the matrix analogue.
* Lower bound is $\Omega(\sqrt T)$, **not** $\Omega(\log T)$: nothing beats $\sqrt T$ rate.

### 16.4 Quick numeric practice patterns
* Given weights and advice: compute $p_t=\frac{\sum\text{weights of experts saying }1}{\sum\text{all weights}}$ and expected loss $=1-p_t$ if $y_t=1$, $p_t$ if $y_t=0$.
* Given $d,T,B$: compute $\eta^*$ and the bound $B\sqrt{2T\ln d}$. E.g. $d=20,\ T=5000,\ B=3$: $\eta^*=\frac13\sqrt{\frac{2\ln20}{5000}}\approx0.0115$; bound $\approx519$.
* Given a $3\times3$ doubly stochastic matrix: run matching + peel; check the $\lambda$'s sum to 1.
* Given $n$: number of actions $n!$, representation dimension $n^2$, at most $n^2+1$ permutations (lecture's bound; algorithm gives $\le n^2$).
* Mistake bound examples: $d=10^6$ $\Rightarrow$ $\lceil\log_2 10^6\rceil=20$.

### 16.5 How the arguments connect (good for conceptual questions)
* *Why regret?* Mistakes can be forced to $T$.
* *Why randomise?* Cover. Deterministic algorithms can be mirrored by the adversary.
* *Why oblivious adversary?* It is the weakest restriction that still lets randomisation help.
* *Why Hedge generalises RWM?* $\ell\in\{0,1\}$ = "expert right/wrong"; $p=w/\Phi$ = RWM's randomisation.
* *Why is $\sqrt T$ optimal?* Coin-flip adversary: any algorithm loses $T/2$ while the best of two experts beats $T/2$ by $\Theta(\sqrt T)$.

---

## 17. Self-test questions with answers

**Q1.** With $d=1000$ experts and a guru, what is the worst-case mistake bound of Majority?
**A.** $\lceil\log_21000\rceil=10$.

**Q2.** Why can't WM have sublinear regret?
**A.** It is deterministic. Cover's adversary (two constant experts, truth = opposite of prediction) forces $R_T\ge T/2$. Also its bound contains a $m^*$ term with constant 1.

**Q3.** State $\Pr(\hat y_t\ne y_t)$ in terms of $p_t=\Pr(\hat y_t=1)$.
**A.** $\lvert p_t-y_t\rvert$.

**Q4.** Weights $(1,\,0.5,\,0.5)$, advice $(0,1,1)$, truth $y_t=0$. Expected loss of RWM?
**A.** $p_t=\frac{0.5+0.5}{2}=0.5$; $y_t=0\Rightarrow$ loss $=p_t=0.5$.

**Q5.** Derive $\epsilon^*$ for RWM.
**A.** Minimise $\frac{\ln d}{\epsilon}+\epsilon T$: $-\frac{\ln d}{\epsilon^2}+T=0\Rightarrow\epsilon^*=\sqrt{\ln d/T}$; value $2\sqrt{T\ln d}$.

**Q6.** Why does the doubling trick work but equal blocks of size $K$ don't?
**A.** Equal blocks: $T/K$ blocks each costing $\Theta(\sqrt K)$ $\Rightarrow$ $\Theta(T/\sqrt K)$, linear in $T$. Doubling: block costs $\sqrt{2^\ell}$ grow geometrically, the sum is dominated by the last block, $\Theta(\sqrt T)$.

**Q7.** Derive Hedge's $\eta^*$ and bound (the slide).
**A.** $C(\eta)=\frac{\ln d}\eta+\frac{\eta TB^2}2$; $C'=0\Rightarrow\eta^*=\frac1B\sqrt{2\ln d/T}$; $C(\eta^*)=\sqrt{2TB^2\ln d}$.

**Q8.** Show that $\min_{q\in\Delta_d}\langle q,L\rangle=\min_iL_i$.
**A.** $\langle q,L\rangle=\sum_iq_iL_i\ge\min_jL_j\sum_iq_i=\min_jL_j$, with equality at the corner $e_{j^*}$.

**Q9.** Why does Hedge never play a corner of the simplex?
**A.** $w_{t,i}=e^{-\eta\sum_s\ell_{s,i}}>0$ always; normalising keeps all coordinates positive.

**Q10.** Why is naive Hedge infeasible for the assignment problem, and what does a permutation-matrix representation change?
**A.** $n!$ actions $\Rightarrow$ $n!$-dimensional distribution. Matrix representation: actions live in $\mathbb R^{n^2}$ with linear loss, but still $n!$ actions. The fix is to maintain a doubly stochastic matrix (point in conv hull) and decompose it with Carathéodory ($\le n^2+1$ permutations).

**Q11.** What are the two "hidden" steps in standard Hedge that become explicit in the combinatorial version?
**A.** Projection (normalisation $\to$ Sinkhorn balancing) and decomposition (sampling $\to$ matching-based Birkhoff decomposition).

**Q12.** Is the decomposition of a doubly stochastic matrix unique? Does it matter?
**A.** Not unique. It doesn't matter for expected loss since $\mathbb E\langle M,L\rangle=\langle P,L\rangle$ regardless of the decomposition.

**Q13.** Complexity of the decomposition algorithm?
**A.** At most $n^2$ iterations (each zeroes $\ge1$ entry), each a poly-time bipartite matching.

**Q14.** Order the following by what they guarantee: Majority, WM, RWM, Hedge.
**A.** Majority: $\log_2d$ mistakes, needs a guru. WM: regret bound but not sublinear. RWM: $O(\sqrt{T\ln d})$ expected regret, $\{0,1\}$ losses. Hedge: same rate for general bounded linear losses.

---

## 18. Lecture slips / things to double-check

The lecture was transcribed automatically and spoken live, so a few things are garbled. I resolved them as follows. Please verify against your own slides if a quiz depends on exact wording.

1. **"Logarithm of $d$" vs $\log_2 d$:** the Majority bound is $\lceil\log_2d\rceil$ (halving). Elsewhere (WM/RWM/Hedge) the natural log $\ln d$ appears.
2. **Final RWM/doubling statement:** the transcript says "$O(\sqrt{T\log T})$" at one point. The derivation (and the later statement "$\sqrt{T\log d}$") gives **$O(\sqrt{T\ln d})$**. $\ln d$ is the number-of-experts term, not $\ln T$.
3. **$\epsilon^*$ typo:** transcript says "$\sqrt{\ln d/d}$"; it should be $\sqrt{\ln d/T}$.
4. **Hedge potential lemma:** the lecture writes $e^x\le1+x+x^2/2$ for $x\le0$ (true). The RWM lecture uses $\ln(1-x)\ge-x-x^2$ on $[0,\frac12]$ (the transcript first says $x^2/2$, then corrects to $x^2$).
5. **Numbers in the $3$-expert example:** transcript drops leading zeros (e.g. "$.04$"). The correct values are $p=0.6$ and loss $0.4$.
6. **Birkhoff decomposition example:** the last lines of the transcript's worked example are garbled; the clean decomposition (weights $0.2,0.2,0.1,0.3,0.2$) is in §14.6 and is checked in the code below.
7. **Carathéodory count:** lecture states $n^2+1$ (dimension $n^2$). The actual dimension of the Birkhoff polytope is $(n-1)^2$, so the sharpest bound is $(n-1)^2+1$ (a known result, **beyond the lecture**). The peeling algorithm gives $\le n^2$. For quizzes use the lecture's numbers.
8. **Lecture timeline:** the lecture notes that the first lectures were recorded ~3 years before the later ones; some topics are repeated for continuity.
9. **WM update detail:** the lecture introduces the update "if a mistake happens"; the standard form and the lower-bound step $w_{k^*}=(1-\epsilon)^{m^*}$ penalise every expert on every round it errs. See the note in §6.1.
10. **"Normalisation = $L_2$ projection":** the lecture says normalising $w$ gives the closest simplex point in the $L_2$ sense, and that Sinkhorn gives the closest doubly stochastic matrix in the Frobenius norm. Strictly, both are **KL (relative-entropy) projections**; the Euclidean projection onto the simplex is *not* plain normalisation. The "normalisation = projection" idea is what matters for the quiz. (Beyond the lecture.)
11. **Lower bound $\mathbb E[\min(N_0,N_1)]=T/2-\Theta(\sqrt T)$** was stated without proof in the lecture, so it may not be examinable beyond the statement.

---

## 19. Runnable verification code

Each block checks a claim from the notes numerically (Majority bound, RWM bound, Hedge bound, Birkhoff decomposition, Sinkhorn, and the worked example). Requires `numpy` and `scipy`. Save as `verify.py` and run `python verify.py`.

```python
"""Runnable sanity checks for Sequential Decision Making, Weeks 1-4 (online learning)."""
import numpy as np
from scipy.optimize import linear_sum_assignment

rng = np.random.default_rng(0)

# ---------- 1. Majority algorithm: mistakes <= ceil(log2 d) when a guru exists ----------
def majority_run(d, T, guru):
    alive = np.ones(d, bool); mistakes = 0
    for _ in range(T):
        x = rng.integers(0, 2, d)          # experts' advice
        y = x[guru]                        # truth = guru's advice (guru is infallible)
        ones = x[alive].sum(); zeros = alive.sum() - ones
        yhat = 1 if ones >= zeros else 0
        mistakes += int(yhat != y)
        alive &= (x == y)                  # keep only experts that were right
    return mistakes

d = 64
worst = max(majority_run(d, 500, rng.integers(d)) for _ in range(200))
print(f"[Majority] worst mistakes over 200 runs = {worst}  (bound ceil(log2 {d}) = {int(np.ceil(np.log2(d)))})")
assert worst <= np.ceil(np.log2(d))

# ---------- 2. Randomized Weighted Majority: regret <= 2 sqrt(T ln d) ----------
def rwm(X, Y, eps):
    T, d = X.shape
    w = np.ones(d); exp_mistakes = 0.0
    for t in range(T):
        p1 = w[X[t] == 1].sum() / w.sum()        # P(predict 1)
        exp_mistakes += (1 - p1) if Y[t] == 1 else p1   # = |p_t - y_t|
        w[X[t] != Y[t]] *= (1 - eps)             # shrink wrong experts
    best = (X != Y[:, None]).sum(0).min()
    return exp_mistakes - best

T, d = 5000, 20
Y = rng.integers(0, 2, T)
noise = rng.uniform(0.2, 0.5, d)                  # every expert is imperfect
X = np.where(rng.random((T, d)) < noise, 1 - Y[:, None], Y[:, None])
eps = np.sqrt(np.log(d) / T)
R = rwm(X, Y, eps)
print(f"[RWM] regret = {R:.1f}  <=  2*sqrt(T ln d) = {2*np.sqrt(T*np.log(d)):.1f}")
assert R <= 2 * np.sqrt(T * np.log(d))

# ---------- 3. Hedge: regret <= B sqrt(2 T ln d) with eta* = (1/B) sqrt(2 ln d / T) ----------
def hedge(L, eta):
    T, d = L.shape
    w = np.ones(d); alg = 0.0
    for t in range(T):
        p = w / w.sum()
        alg += p @ L[t]
        w *= np.exp(-eta * L[t])
    return alg - L.sum(0).min()

B = 3.0
L = rng.uniform(0, B, (T, d)); L[:, 7] *= 0.8     # expert 7 is slightly better on average
eta = np.sqrt(2 * np.log(d) / T) / B
R = hedge(L, eta)
print(f"[Hedge] regret = {R:.1f}  <=  B*sqrt(2 T ln d) = {B*np.sqrt(2*T*np.log(d)):.1f}")
assert R <= B * np.sqrt(2 * T * np.log(d))

# ---------- 4. Birkhoff-von Neumann decomposition (matching + peel off) ----------
def bvn(A, tol=1e-12):
    A = A.copy(); n = len(A); parts = []
    while A.max() > tol:
        cost = np.where(A > tol, 0.0, 1.0)         # 0 on edges of the support graph
        r, c = linear_sum_assignment(cost)         # a perfect matching on the support
        assert cost[r, c].sum() == 0, "no matching -> A was not a scaled doubly-stochastic matrix"
        lam = A[r, c].min()                        # smallest edge in the matching
        P = np.zeros((n, n)); P[r, c] = 1
        parts.append((lam, P)); A -= lam * P       # zero out >=1 entry per round
    return parts

A = np.array([[.2, .3, .5], [.4, .4, .2], [.4, .3, .3]])
parts = bvn(A)
print(f"[BvN] {len(parts)} permutation matrices, weights = {[round(float(l),2) for l,_ in parts]}")
assert np.allclose(sum(l * P for l, P in parts), A) and abs(sum(l for l, _ in parts) - 1) < 1e-9

# ---------- 5. Sinkhorn balancing: alternate row / column normalisation ----------
def sinkhorn(W, iters=1000):
    M = W.copy()
    for _ in range(iters):
        M /= M.sum(1, keepdims=True)               # rows sum to 1
        M /= M.sum(0, keepdims=True)               # columns sum to 1
    return M

W = rng.random((5, 5)) + 0.1
M = sinkhorn(W)
print("[Sinkhorn] row sums:", M.sum(1).round(6), " col sums:", M.sum(0).round(6))
assert np.allclose(M.sum(1), 1, atol=1e-6) and np.allclose(M.sum(0), 1, atol=1e-6)

# ---------- 6. Worked example from lecture: weights (0.2, 0.3, 0.25), advice (1,0,1), truth 1 ----------
w = np.array([0.2, 0.3, 0.25]); x = np.array([1, 0, 1])
p1 = w[x == 1].sum() / w.sum()
print(f"[Example] P(predict 1) = {p1:.2f}; expected loss when y=1 is {1-p1:.2f}")
print("All checks passed.")

```

**Expected output**

```text
[Majority] worst mistakes over 200 runs = 5  (bound ceil(log2 64) = 6)
[RWM] regret = 100.6  <=  2*sqrt(T ln d) = 244.8
[Hedge] regret = 264.3  <=  B*sqrt(2 T ln d) = 519.2
[BvN] 5 permutation matrices, weights = [0.2, 0.1, 0.3, 0.2, 0.2]
[Sinkhorn] row sums: [1. 1. 1. 1. 1.]  col sums: [1. 1. 1. 1. 1.]
[Example] P(predict 1) = 0.60; expected loss when y=1 is 0.40
All checks passed.
```

---
*End of notes.*
