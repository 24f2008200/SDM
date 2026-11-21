# Sequential Decision Making — Week 8 Notes
**Contextual Bandits, Bandits with Expert Advice (mixing algorithm), and Why Pure Exploration / Pure Greedy Fail in Stochastic Bandits**

> Continuation of Weeks 1–7. Built from lecture transcript L46–51.
> Math is GitHub-flavoured LaTeX (`$...$`, `$$...$$`).
> Notation: $d$ = number of arms, $T$ = rounds, $I_t$ = arm played, $\ell_t\in[0,1]^d$ = loss vector, $\hat\ell_t$ = importance-weighted estimate, $\mathcal F$ = set of contexts ($K=\lvert\mathcal F\rvert$), $L$ = number of experts, $a_{t,l}\in\Delta_d$ = advice of expert $l$, $q_t\in\Delta_L$ = belief over experts, $\eta$ = learning rate, $\ln$ = natural log.
>
> **About the source file.** L46–51 contains four segments: (1) contextual bandits + the first (naive) algorithm for bandits with expert advice, (2) a **repeat of the Randomized Weighted Majority lecture** (already covered in the Weeks 1–4 notes, so skipped here), (3) the **mixing algorithm** for bandits with expert advice, (4) the first **stochastic-bandits** lecture. Items that are *not* in the lecture (my own derivations / known results) are marked **(beyond the lecture)**.

---

## Table of Contents
1. [One-page summary](#1-one-page-summary)
2. [Recap and roadmap](#2-recap-and-roadmap)
3. [Contextual multi-armed bandits (CMAB)](#3-contextual-multi-armed-bandits-cmab)
4. [CMAB-EXP3: one EXP3 per context](#4-cmab-exp3-one-exp3-per-context)
5. [Bandits with expert advice: setting](#5-bandits-with-expert-advice-setting)
6. [Naive algorithm: EXP3 over experts](#6-naive-algorithm-exp3-over-experts)
7. [Better algorithm: mix first, then update all experts](#7-better-algorithm-mix-first-then-update-all-experts)
8. [Stochastic bandits: setup and regret decomposition](#8-stochastic-bandits-setup-and-regret-decomposition)
9. [Pure exploration fails](#9-pure-exploration-fails)
10. [Pure exploitation (greedy) fails](#10-pure-exploitation-greedy-fails)
11. [Where next](#11-where-next)
12. [Formula table](#12-formula-table)
13. [Quiz preparation tips](#13-quiz-preparation-tips)
14. [Self-test questions with answers](#14-self-test-questions-with-answers)
15. [Lecture slips / things to double-check](#15-lecture-slips--things-to-double-check)
16. [Runnable verification code](#16-runnable-verification-code)

---

## 1. One-page summary

| Topic | Key facts |
|---|---|
| **Where we are** | Adversarial bandits solved by EXP3 (Hedge + unbiased importance-weighted loss estimate): $R_T=O(\sqrt{Td\ln d})$. Week 8 extends the **setting** (context, expert advice) and starts **stochastic** bandits |
| **CMAB protocol** | Round $t$: receive context $f_t\in\mathcal F$ ($\lvert\mathcal F\rvert=K$, finite) $\to$ pull arm $I_t$ $\to$ see only $\ell_{t,I_t}$ |
| **Policy** | $\pi:\mathcal F\to[d]$ (context $\to$ arm). Fixed-arm policies are special cases; there are $d^K$ policies in total |
| **CMAB regret** | $R_T=\mathbb E\sum_t\ell_{t,I_t}-\min_{\pi\in\Pi}\sum_t\ell_{t,\pi(f_t)}$ — compare with the **best policy**, a much stronger comparator than the best single arm |
| **CMAB-EXP3** | Run a **separate EXP3 for each context** (invoked only when that context appears) |
| **CMAB regret** | $\displaystyle R_T\le\sum_f\sqrt{2T_fd\ln d}\le\sqrt{2\,T\,K\,d\ln d}=O\big(\sqrt{T\lvert\mathcal F\rvert d\ln d}\big)$ via **Cauchy–Schwarz** ($T_f$ = # rounds with context $f$, $\sum_fT_f=T$) |
| **Limitation** | Bad if $K$ is huge (contexts treated as independent; no learning across contexts). Fix needs **structured** loss–context relationships (later) |
| **Bandits with expert advice** | $L$ experts each give a **distribution** $a_{t,l}\in\Delta_d$ over arms; compare with the best expert: $R_T=\mathbb E\sum_t\ell_{t,I_t}-\min_l\sum_t\langle a_{t,l},\ell_t\rangle$ |
| **Naive algorithm** | Maintain $q_t\in\Delta_L$; draw an expert $e_t\sim q_t$, then an arm from $a_{t,e_t}$; run **EXP3 over experts** with $\hat y_{t,j}=\frac{\ell_{t,I_t}}{q_{t,j}}\mathbf 1[e_t=j]$. Only the **drawn** expert is updated |
| **Mixing algorithm** (EXP4-type) | **First mix** $p_t=\sum_lq_{t,l}a_{t,l}\in\Delta_d$, draw $I_t\sim p_t$, build $\hat\ell_{t,i}=\frac{\ell_{t,i}}{p_{t,i}}\mathbf 1[I_t=i]$, then **map to experts**: $\hat y_{t,l}=\langle a_{t,l},\hat\ell_t\rangle=a_{t,l}(I_t)\,\frac{\ell_{t,I_t}}{p_{t,I_t}}$; feed to Hedge. **Every expert is updated** each round |
| **Stochastic bandits** | Arm $i$ gives i.i.d. losses with mean $\mu_i$; gap $\Delta_i=\mu_i-\mu_{i^*}$. $R_T=\sum_i\Delta_i\,\mathbb E[N_i(T)]$ |
| **Pure exploration** (uniform random arm every round) | $R_T=\frac Td\sum_i\Delta_i$ = **linear** |
| **Pure greedy** (pull each arm once, then commit to the best empirical arm) | With probability $0.16$ (2 Bernoulli arms, means $0.4$ vs $0.6$) it commits to the bad arm: $R_T\ge0.16\times0.2\times(T-2)=\Omega(T)$ = **linear** |
| **Moral** | Algorithms must **balance exploration and exploitation** (neither extreme works) |

**Five ideas to remember**
1. **Stronger comparator** (best *policy*) $\Rightarrow$ harder problem, but contexts from a finite set can be solved by **independent copies** of EXP3.
2. The CMAB bound's $\sqrt K$ comes from **Cauchy–Schwarz**: $\sum_f\sqrt{T_f}\le\sqrt K\sqrt{\sum_fT_f}$.
3. In bandits with expert advice, **share information across experts through the arm**: one observed arm loss says something about every expert (weighted by how much each expert bet on that arm).
4. The $L\to d$ idea: naive treatment pays in the number of experts $L$; mixing first should pay in the number of arms $d$ (analysis was deferred to the next lecture).
5. **Neither pure exploration nor pure exploitation gives sublinear regret** in stochastic bandits: a regret formula $R_T=\sum_i\Delta_iE[N_i]$ tells you what to control (the number of pulls of bad arms).

---

## 2. Recap and roadmap

* **Adversarial bandits:** the adversary fixes an (oblivious) loss matrix; you see only $\ell_{t,I_t}$. EXP3 = **Exp**onentiate, **Exp**lore, **Exp**loit. It uses Hedge as a black box, fed with an **unbiased importance-weighted estimate** of the loss vector, and enjoys $R_T=O(\sqrt{Td\ln d})$: sublinear regret is possible even though nothing is known about unplayed arms and the future need not resemble the past.
* **This week:** two **practical extensions of the setting** where EXP3 is used as a black box (or as inspiration), and then a first look at **stochastic** bandits:
  1. contextual bandits (easy: reuse EXP3 per context);
  2. bandits with expert advice (harder: the experts' advice are probability distributions);
  3. stochastic bandits (what goes wrong with the two extreme strategies).

---

## 3. Contextual multi-armed bandits (CMAB)

### 3.1 Setting
* Same bandit game, plus a **context** (side information) in every round: a user arrives at a portal / a patient walks into a hospital with some features.
* **Finite context set** $\mathcal F=\{f_1,\dots,f_K\}$ ($K=\lvert\mathcal F\rvert$). Examples: users binned into 10 types; patients described by blood pressure $\in\{\text{low, medium, high}\}$ and sugar $\in\{\text{low, medium, high}\}$ — a bunch of features each taking finitely many values still gives a **finite** (possibly large) set of contexts.
* Real-valued features can be **binned** by thresholds (e.g. height $>180$ cm vs $\le180$ cm), keeping $\mathcal F$ finite.
* Arms are unchanged: $d$ advertisements / medications / movies.

### 3.2 Protocol
For $t=1,\dots,T$:
1. Receive context $f_t\in\mathcal F$ from the environment.
2. Pull arm $I_t\in[d]$.
3. Suffer (and observe) only $\ell_{t,I_t}$ (partial feedback, as before).

**Applications.** Context = user features; arms = products (movies, books, phones). Context = symptoms/test results; arms = medications.

### 3.3 Why the usual regret is the wrong yardstick
The standard regret compares with the **best single fixed arm**:
$$\mathbb E\sum_t\ell_{t,I_t}-\min_{j\in[d]}\sum_t\ell_{t,j}.$$
With different user types, no single movie/ad is good for everyone (e.g. $10\%$ like comedy, $10\%$ action, $10\%$ romance, ...). Then *every* fixed arm has large loss, so we would be competing against a **weak opponent**. We need a **stronger comparator class** that actually uses the context. (The lecture asked students to pause and come up with it.)

### 3.4 Policies and the CMAB regret
A **policy** is a map
$$\pi:\mathcal F\to[d]\qquad(\text{"if you see context }f\text{, play arm }\pi(f)\text{"}).$$
$\Pi$ = set of all policies. Playing policy $\pi$ in every round costs $\sum_{t}\ell_{t,\pi(f_t)}$.

$$
\boxed{\ R_T=\mathbb E\Big[\sum_{t=1}^T\ell_{t,I_t}\Big]-\min_{\pi\in\Pi}\sum_{t=1}^T\ell_{t,\pi(f_t)}\ }
$$

* Goal: be competitive with the **best policy in hindsight**.
* **Strictly harder** than the standard regret: "always play arm $j$" is itself a policy (there are $d$ such constant policies), but $\Pi$ contains many more: e.g. "comedy for type-1 users, action for type-2 users". Hence the minimum over $\Pi$ is no larger than (and usually much smaller than) the best-fixed-arm loss.
* $\lvert\Pi\rvert=d^K$ (each of the $K$ contexts can be mapped to any of the $d$ arms), so the class is huge, yet we want sublinear regret against all of it.

---

## 4. CMAB-EXP3: one EXP3 per context

### 4.1 The idea
Because $\Pi$ contains **every** map from contexts to arms, **nothing learned about one context says anything about another**. Partition the $T$ rounds by context; the rounds with context $f$ form an independent bandit problem. So:

> **Run a separate EXP3 instance for each context.**

### 4.2 Algorithm
For $t=1,\dots,T$:
1. Receive $f_t\in\mathcal F$.
2. **Invoke the EXP3 instance of $f_t$**; it returns its current distribution $p_t\in\Delta_d$.
3. Play $I_t\sim p_t$, suffer $\ell_{t,I_t}$.
4. Feed the loss back to **that same instance only**, which updates its weights.

An instance is touched only in rounds where its context appears. Each instance is competitive with the best *arm for its own context*, and putting the per-context best arms together gives the best *policy*.

### 4.3 Regret analysis
**Step 1: split by context.** Since the best policy chooses the best arm for each context separately,
$$
\min_{\pi\in\Pi}\sum_t\ell_{t,\pi(f_t)}=\sum_{f\in\mathcal F}\min_{j\in[d]}\sum_{t:\,f_t=f}\ell_{t,j}
$$
(checked by brute force in the code), and by linearity of expectation
$$
R_T=\sum_{f\in\mathcal F}\Bigg(\mathbb E\sum_{t:f_t=f}\ell_{t,I_t}-\min_{j}\sum_{t:f_t=f}\ell_{t,j}\Bigg).
$$

**Step 2: each bracket is the regret of one EXP3.** Let $T_f$ = number of rounds with context $f$ (at most $T$; $\sum_fT_f=T$ since every round has exactly one context). Within those rounds the instance plays a standard bandit game, so its bracket is at most the EXP3 bound for $T_f$ rounds:
$$
R_T\ \le\ \sum_{f\in\mathcal F}\sqrt{2\,T_f\,d\ln d}\ =\ \sqrt{2d\ln d}\ \sum_{f\in\mathcal F}\sqrt{T_f}.
$$

**Step 3: convert to a bound in $T$ (Cauchy–Schwarz).** Write $\sum_f\sqrt{T_f}=\langle\mathbf 1,v\rangle$ with $\mathbf 1\in\mathbb R^K$ the all-ones vector and $v=(\sqrt{T_{f_1}},\dots,\sqrt{T_{f_K}})$. Cauchy–Schwarz, $\langle a,b\rangle\le\lVert a\rVert\lVert b\rVert$, gives
$$
\sum_f\sqrt{T_f}\ \le\ \underbrace{\lVert\mathbf 1\rVert}_{\sqrt K}\cdot\underbrace{\lVert v\rVert}_{\sqrt{\sum_fT_f}=\sqrt T}=\sqrt{KT}.
$$

$$
\boxed{\ R_T(\text{CMAB-EXP3})\ \le\ \sqrt{2\,T\,K\,d\,\ln d}=O\Big(\sqrt{T\,\lvert\mathcal F\rvert\,d\ln d}\Big)\ }
$$

(The lecture writes the bound without the constant 2.) The only new factor compared to EXP3 is $\sqrt{\lvert\mathcal F\rvert}$.

> **(beyond the lecture)** Per-context learning rates $\eta_f=\sqrt{2\ln d/(T_fd)}$ require knowing $T_f$ in advance; if not known, use the doubling trick inside each instance, which changes only constants.

### 4.4 When is this a good algorithm?
* **Good** when $K=\lvert\mathcal F\rvert$ is moderate (you maintain $K$ copies of EXP3, and pay $\sqrt K$).
* **Bad** when $K$ is huge. E.g. $10$ features with $10$ bins each give $10^{10}$ contexts: too many instances to maintain and a vacuous bound.
* **Reason:** the algorithm **treats contexts as unrelated**. Playing arm $i$ for one user and seeing a small loss tells nothing about similar users. This independence is built into the **comparator class** (all context-to-arm maps allowed). If we instead assumed structure (e.g. loss is a *linear* or quadratic function of the context features), the comparator class would be much smaller and one could do better: **later lectures**.

> **(beyond the lecture)** Observation: $\ln\lvert\Pi\rvert=K\ln d$, so $\sqrt{2TKd\ln d}=\sqrt{2\,T\,d\,\ln\lvert\Pi\rvert}$, which is exactly the bound the mixing algorithm of §7 gets (see §7.5) when *all* $d^K$ policies are used as experts, while CMAB-EXP3 needs only $K$ instances rather than $d^K$ weights.

---

## 5. Bandits with expert advice: setting

### 5.1 The twist on the expert problem
Back to the first problem of the course (experts), but now:
* **Arms** $=$ stocks ($d$ of them, e.g. $d$ companies).
* **Experts** $=$ mutual-fund managers ($L$ of them). An expert does **not** say "invest in stock 3"; each gives a **probability distribution over the arms** saying how to split your money.
* You want to be as good as the **best expert in hindsight**, but you only observe the loss of the arm you played (bandit feedback).

### 5.2 Protocol
For $t=1,\dots,T$:
1. Receive advice $a_{t,1},\dots,a_{t,L}$, each $a_{t,l}\in\Delta_d$ ($L$ probability distributions over $d$ arms).
2. Pick an arm $I_t$.
3. Suffer/observe only $\ell_{t,I_t}$.

### 5.3 Why it is not a contextual bandit
One could try to treat the advice vectors as a context, but they are **points of the simplex** (continuous). Binning the simplex for each of $L$ experts would give far too many bins, so the finite-context machinery of §4 does not apply directly.

### 5.4 Regret
How good is expert $l$ in round $t$? If you followed its advice, you would draw an arm $I\sim a_{t,l}$ and suffer $\ell_{t,I}$, i.e. the **expected loss**
$$
\mathbb E_{I\sim a_{t,l}}[\ell_{t,I}]=\sum_{i=1}^da_{t,l}(i)\,\ell_{t,i}=\langle a_{t,l},\ell_t\rangle .
$$
(Example: advice "$30\%,30\%,30\%,10\%$ across four stocks" has expected loss $0.3\ell_1+0.3\ell_2+0.3\ell_3+0.1\ell_4$.) We can reason about this in expectation even though $\ell_t$ is unknown to us.

$$
\boxed{\ R_T=\mathbb E\Big[\sum_{t=1}^T\ell_{t,I_t}\Big]-\min_{l\in[L]}\sum_{t=1}^T\langle a_{t,l},\ell_t\rangle\ }
$$

You "regret" not having followed the best fund manager for all $T$ rounds.

---

## 6. Naive algorithm: EXP3 over experts

**Big idea.** Treat the **experts as the arms** and play a **meta-EXP3** over experts: maintain a distribution $q_t\in\Delta_L$ over experts ("how much do I trust expert $l$?"), initialised uniformly $q_{1,l}=\frac1L$.

For $t=1,\dots,T$:
1. Receive $a_{t,1},\dots,a_{t,L}\in\Delta_d$.
2. **Expert-picking step:** draw $e_t\sim q_t$.
3. **Arm-picking step:** draw $I_t\sim a_{t,e_t}$ (the arm distribution recommended by the chosen expert) and play it. Observe $\ell_{t,I_t}$.
4. **Fake expert loss** (importance weighting over experts):
$$
\hat y_{t,j}=\frac{\ell_{t,I_t}}{q_{t,j}}\,\mathbf 1[e_t=j]\qquad(j\in[L]).
$$
5. Feed $\hat y_t\in\mathbb R^L$ to Hedge to obtain $q_{t+1}\in\Delta_L$.

* It is **unbiased** for the expert's expected loss: given $e_t=j$ (probability $q_{t,j}$), $I_t\sim a_{t,j}$, so $\mathbb E[\hat y_{t,j}]=q_{t,j}\cdot\frac{\langle a_{t,j},\ell_t\rangle}{q_{t,j}}=\langle a_{t,j},\ell_t\rangle$.
* **Drawback ("naive"):** only the **drawn** expert $e_t$ receives a non-zero fake loss; the other $L-1$ experts learn nothing, even though the arm loss we observed is relevant to all of them. When $L$ is large (the case of interest, $L>d$) this is wasteful.
* *(Consequence, my derivation — beyond the lecture)*: this is EXP3 with $L$ "arms", so its regret scales like $\sqrt{2\,T\,L\ln L}$.

---

## 7. Better algorithm: mix first, then update all experts

### 7.1 The idea
Do **not** pick an expert first. Instead:
1. **Mix** all experts' distributions with the algorithm's belief $q_t$ to get one distribution over **arms**;
2. play an arm from it;
3. build an **unbiased estimate of the loss vector over arms** (as in EXP3);
4. **map** that arm-loss estimate to a loss vector over **experts**, and update $q_t$ with Hedge.

The algorithm still keeps a distribution over experts (the comparator is the best expert), but one observed arm loss now informs **all** experts. (The lecture introduces the algorithm without a name; it is the standard **EXP4**-type algorithm.)

### 7.2 Algorithm
Initialise $q_{1,l}=\frac1L$. For $t=1,\dots,T$:

1. Receive $a_{t,1},\dots,a_{t,L}\in\Delta_d$.
2. **Mixing (belief over arms):**
$$
\boxed{\ p_{t,i}=\sum_{l=1}^Lq_{t,l}\,a_{t,l}(i)\quad(i\in[d]),\qquad\text{i.e. }p_t=\sum_lq_{t,l}a_{t,l}\in\Delta_d\ }
$$
(a convex combination of the experts' arm-distributions, weighted by trust).
3. Draw $I_t\sim p_t$; observe $\ell_{t,I_t}$.
4. **Fake loss over arms** (standard importance weighting):
$$
\hat\ell_{t,i}=\frac{\ell_{t,i}}{p_{t,i}}\mathbf 1[I_t=i]\qquad(i\in[d]).
$$
5. **Fake loss over experts** (pretend $\hat\ell_t$ is the true $\ell_t$ and take the expert's expected loss):
$$
\boxed{\ \hat y_{t,l}=\langle a_{t,l},\hat\ell_t\rangle=a_{t,l}(I_t)\cdot\frac{\ell_{t,I_t}}{p_{t,I_t}}\qquad(l\in[L])\ }
$$
6. Feed $\hat y_t\in\mathbb R^L$ to Hedge to get $q_{t+1}\in\Delta_L$.

**Why $\hat y_{t,l}=a_{t,l}(I_t)\,\ell_{t,I_t}/p_{t,I_t}$:** $\hat\ell_t$ has a single non-zero entry (at $I_t$), so the dot product with $a_{t,l}$ keeps just one term. **Every** expert gets a non-zero value, scaled by how much that expert bet on the arm that was played.

**Unbiasedness.** $\mathbb E[\hat\ell_t\mid p_t]=\ell_t$ (EXP3 argument), and $\hat y_t=A_t\hat\ell_t$ is linear in $\hat\ell_t$, so $\mathbb E[\hat y_{t,l}]=\langle a_{t,l},\ell_t\rangle$, the true expected loss of expert $l$ (verified in the code).

### 7.3 Worked example (lecture's numbers, reconstructed)
Three experts, three arms:
$$
a_1=(0.9,\,0.05,\,0.05),\quad a_2=(0.8,\,0.1,\,0.1),\quad a_3=(0.1,\,0.8,\,0.1),\qquad q_t=(0.4,\,0.4,\,0.2).
$$
Mixing:
$$
p_{t,1}=0.9(0.4)+0.8(0.4)+0.1(0.2)=0.70,\quad
p_{t,2}=0.05(0.4)+0.1(0.4)+0.8(0.2)=0.22,\quad
p_{t,3}=0.05(0.4)+0.1(0.4)+0.1(0.2)=0.08 .
$$
Suppose arm 1 is drawn and the observed loss is $0.1$:
$$
\hat\ell_t=\Big(\tfrac{0.1}{0.7},0,0\Big)=(0.143,0,0).
$$
Fake expert losses:

| | Expert 1 | Expert 2 | Expert 3 |
|---|---|---|---|
| **Mixing algorithm** $\hat y_{t,l}=a_{t,l}(1)\cdot0.143$ | $0.9\times0.143=0.129$ | $0.8\times0.143=0.114$ | $0.1\times0.143=0.014$ |
| **Naive** (expert 1 happened to be drawn, $q_{t,1}=0.4$) | $0.1/0.4=0.25$ | $0$ | $0$ |

The mixing vector has **informative entries everywhere** (experts that bet heavily on arm 1 receive more loss, expert 3 who barely bet on it receives almost none). The naive vector is zero except one entry. This richer information is why we hope for a better regret bound (which should involve $d$ instead of $L$).

### 7.4 Regret
The lecture ends here: **"we will come back next time and analyze it"**. So the regret bound is *not* in this lecture.

### 7.5 (beyond the lecture) The expected bound, derived in my own steps
Apply Hedge's inequality (valid for any non-negative loss vectors) over the **$L$ experts**:
$$
\sum_t\langle q_t,\hat y_t\rangle-\sum_t\hat y_{t,l}\le\frac{\ln L}{\eta}+\frac\eta2\sum_t\sum_lq_{t,l}\hat y_{t,l}^{\,2}.
$$
* $\langle q_t,\hat y_t\rangle=\sum_lq_{t,l}\langle a_{t,l},\hat\ell_t\rangle=\langle p_t,\hat\ell_t\rangle$, whose expectation is $\langle p_t,\ell_t\rangle$ = the algorithm's expected loss.
* $\mathbb E\hat y_{t,l}=\langle a_{t,l},\ell_t\rangle$ = expert's expected loss.
* Second-moment term: by Jensen (each $a_{t,l}$ is a distribution), $\langle a_{t,l},\hat\ell_t\rangle^2\le\langle a_{t,l},\hat\ell_t^{\,2}\rangle$, so $\sum_lq_{t,l}\hat y_{t,l}^2\le\langle p_t,\hat\ell_t^{\,2}\rangle$, whose expectation is $\sum_i\ell_{t,i}^2\le d$ (the EXP3 cancellation).

Hence $R_T\le\frac{\ln L}{\eta}+\frac{\eta Td}{2}$, and with $\eta^*=\sqrt{\frac{2\ln L}{Td}}$:
$$
R_T\le\sqrt{2\,T\,d\,\ln L}.
$$
Compare with the naive algorithm's $\sqrt{2TL\ln L}$: the dependence on the number of experts drops from $\sqrt L$ to $\sqrt{\ln L}$ and instead the number of **arms** appears. *(Numerically checked in the code; the official analysis is the next lecture, so treat constants as mine.)*

---

## 8. Stochastic bandits: setup and regret decomposition

(From the previous lecture, recalled at the start of the last segment.)

* **Stochastic bandits:** each arm $i$ produces **i.i.d.** losses/rewards with an unknown mean (arms are independent random variables). The **best arm** is the one with the smallest expected loss $\mu_{i^*}=\min_i\mu_i$ (or largest expected reward).
* **Suboptimality gap:** $\Delta_i=\mu_i-\mu_{i^*}\ge0$.
* **Regret decomposition:** because the arms are i.i.d., instead of summing over rounds we sum over arms:

$$
\boxed{\ R_T=\sum_{i=1}^d\Delta_i\ \mathbb E\big[N_i(T)\big]\ }\qquad N_i(T)=\sum_{t=1}^T\mathbf 1[I_t=i]\ \ (\text{number of pulls of arm }i)
$$

Two factors per arm: **how bad** the arm is ($\Delta_i$) times **how often** we play it.
* **Consequence:** to get a regret bound it is enough to bound the **expected number of pulls of every suboptimal arm**.
* **Exploration vs exploitation:**
  * **Explore:** keep pulling arms that currently look bad, since randomness may have made them look bad while their true loss is low.
  * **Exploit:** keep playing arms already known to be good.
  * A good algorithm must **balance** both. Before building such algorithms, the lecture shows that each extreme fails.

---

## 9. Pure exploration fails

**Algorithm:** in every round play an arm **uniformly at random** ($p_i=\frac1d$), never using the observed losses.

For arm $i$: $\mathbb E[N_i(T)]=\sum_{t=1}^T\Pr[I_t=i]=\frac Td$. So the regret is **exactly**
$$
\boxed{\ R_T=\sum_i\Delta_i\cdot\frac Td=T\cdot\underbrace{\frac1d\sum_{i=1}^d\Delta_i}_{\text{constant}}\ }
$$

* If even **one** arm is strictly suboptimal ($\Delta_i>0$, however small), the constant is positive, so $R_T/T\to\text{const}>0$: **linear regret**.
* Obvious intuitively; here it is also shown mathematically: ignoring the information that some arms are better means no sublinear regret.

---

## 10. Pure exploitation (greedy) fails

**Algorithm (pure greedy):** pull each arm **once**; then **forever play the arm with the smallest empirical loss** (no further exploration).

To show a bad algorithm, one counterexample suffices (a good algorithm must work for *every* instance).

### 10.1 Counterexample: two Bernoulli arms
* $d=2$. Arm 1: loss $1$ with probability $0.4$ (else $0$), so $\mu_1=0.4$ (**best arm**). Arm 2: loss $1$ with probability $0.6$, so $\mu_2=0.6$. Gap $\Delta_2=0.2$.
* The two initial pulls are independent. **Bad event:** arm 1 returns $1$ **and** arm 2 returns $0$:
$$
\Pr[\text{bad event}]=0.4\times0.4=0.16 .
$$
* In the bad event, arm 2 has the smaller empirical loss ($0<1$), so greedy **commits to arm 2 forever**: it plays the suboptimal arm in all of the remaining $T-2$ rounds.
* **Regret in that event** $\approx\Delta_2\,(T-2)=0.2(T-2)$.

### 10.2 Regret lower bound
$$
R_T\ \ge\ 0.16\times0.2\times(T-2)=0.032\,(T-2)\qquad\Longrightarrow\qquad\frac{R_T}{T}\not\to0 .
$$
(Ties broken in favour of arm 1: the other three outcomes lead greedy to arm 1 and zero further regret; the lower bound needs only the bad event, which has probability $0.16$.) Exact value in this instance: $\mathbb E[R_T]=0.032(T-2)+0.2$ (the extra $0.2$ is the forced first pull of arm 2).

In terms of pulls: $\mathbb E[N_1(T)]=0.84(T-1)+0.16\cdot1$, so the **suboptimal arm is pulled linearly often** (exercise left by the lecture).

**Lesson.** A single unlucky early sample can lock greedy onto a bad arm. A *constant* probability of a *linear* regret event gives *linear expected regret*. To fix it, collect **enough information before committing**; the amount of exploration must **depend on $T$** so that the chance of committing to a bad arm is small enough.

---

## 11. Where next

* **From pure exploration:** use the observed losses to **change the sampling distribution** (explore, but adaptively).
* **From pure greedy:** explore each arm **more than once** (e.g. $10$ or $50$ times, or a number depending on $T$) before exploiting; analyse how many pulls make the bad event unlikely enough.
* **Next lectures:** the analysis of the expert-advice mixing algorithm, and algorithms for stochastic bandits that balance exploration and exploitation explicitly or implicitly.

---

## 12. Formula table

| # | Concept | Formula | Notes |
|---|---|---|---|
| 1 | CMAB protocol | receive $f_t\in\mathcal F$; play $I_t$; see $\ell_{t,I_t}$ | $K=\lvert\mathcal F\rvert<\infty$ |
| 2 | Policy | $\pi:\mathcal F\to[d]$; $\lvert\Pi\rvert=d^K$ | constants $\subset\Pi$ |
| 3 | CMAB regret | $R_T=\mathbb E\sum_t\ell_{t,I_t}-\min_{\pi\in\Pi}\sum_t\ell_{t,\pi(f_t)}$ | best policy |
| 4 | Per-context split | $\min_\pi\sum_t\ell_{t,\pi(f_t)}=\sum_f\min_j\sum_{t:f_t=f}\ell_{t,j}$ | policy picks arms independently |
| 5 | Context counts | $T_f=\#\{t:f_t=f\}$, $\sum_fT_f=T$ | |
| 6 | CMAB-EXP3 bound | $R_T\le\sum_f\sqrt{2T_fd\ln d}$ | one EXP3 per context |
| 7 | Cauchy–Schwarz step | $\sum_f\sqrt{T_f}=\langle\mathbf 1,v\rangle\le\sqrt K\sqrt T$ | $v_f=\sqrt{T_f}$ |
| 8 | CMAB final | $R_T\le\sqrt{2TKd\ln d}=O(\sqrt{T\lvert\mathcal F\rvert d\ln d})$ | |
| 9 | Expert advice | $a_{t,l}\in\Delta_d$, $l\in[L]$ | distributions over arms |
| 10 | Expert's expected loss | $\langle a_{t,l},\ell_t\rangle=\sum_ia_{t,l}(i)\ell_{t,i}$ | |
| 11 | BEA regret | $R_T=\mathbb E\sum_t\ell_{t,I_t}-\min_l\sum_t\langle a_{t,l},\ell_t\rangle$ | |
| 12 | Naive: draws | $e_t\sim q_t$, $I_t\sim a_{t,e_t}$ | meta-EXP3 over experts |
| 13 | Naive: fake loss | $\hat y_{t,j}=\frac{\ell_{t,I_t}}{q_{t,j}}\mathbf 1[e_t=j]$ | only drawn expert updated |
| 14 | Mixing | $p_t=\sum_lq_{t,l}a_{t,l}$, i.e. $p_{t,i}=\sum_lq_{t,l}a_{t,l}(i)$ | belief over arms |
| 15 | Arm fake loss | $\hat\ell_{t,i}=\frac{\ell_{t,i}}{p_{t,i}}\mathbf 1[I_t=i]$ | unbiased |
| 16 | Expert fake loss | $\hat y_{t,l}=\langle a_{t,l},\hat\ell_t\rangle=a_{t,l}(I_t)\frac{\ell_{t,I_t}}{p_{t,I_t}}$ | all experts updated; $\mathbb E\hat y_{t,l}=\langle a_{t,l},\ell_t\rangle$ |
| 17 | Hedge on experts | $q_{t+1}\leftarrow\text{Hedge}(q_t,\hat y_t)$, i.e. $q_{t+1,l}\propto q_{t,l}e^{-\eta\hat y_{t,l}}$ | |
| 18 | Mixing bound *(beyond lecture)* | $R_T\le\frac{\ln L}{\eta}+\frac{\eta Td}2$, $\eta^*=\sqrt{\frac{2\ln L}{Td}}$, $R_T\le\sqrt{2Td\ln L}$ | next lecture's analysis |
| 19 | Naive bound *(beyond lecture)* | $R_T\le\sqrt{2TL\ln L}$ | EXP3 over $L$ experts |
| 20 | Stochastic gap | $\Delta_i=\mu_i-\mu_{i^*}$ | losses: $\mu_{i^*}=\min_i\mu_i$ |
| 21 | Regret decomposition | $R_T=\sum_i\Delta_i\mathbb E[N_i(T)]$ | $N_i$ = # pulls of $i$ |
| 22 | Pure exploration | $\mathbb E[N_i]=T/d$; $R_T=\frac Td\sum_i\Delta_i$ | linear |
| 23 | Greedy bad event | $\Pr=0.4\cdot0.4=0.16$ | means $0.4$ vs $0.6$ |
| 24 | Greedy regret | $R_T\ge0.16\cdot0.2\cdot(T-2)=0.032(T-2)$ | linear |

---

## 13. Quiz preparation tips

### 13.1 Definitions to state precisely
* Contextual bandit protocol; context set $\mathcal F$; **policy**; policy class $\Pi$; CMAB regret.
* Bandits with expert advice: what each expert outputs (a **distribution** over arms), expert's expected loss, regret against the best expert.
* Stochastic bandits: arm means $\mu_i$, gaps $\Delta_i$, $N_i(T)$, the regret decomposition.
* Pure exploration, pure greedy/exploitation.

### 13.2 Derivations likely to be asked
1. **CMAB-EXP3 regret:** per-context split $\to$ EXP3 bound per context $\to$ Cauchy–Schwarz $\to\sqrt{2TKd\ln d}$.
2. **Why the best policy decomposes per context** (policy chooses each context's arm independently).
3. **Mixing algorithm steps** (mix $\to$ draw arm $\to$ $\hat\ell$ over arms $\to$ $\hat y=A\hat\ell$ over experts $\to$ Hedge) and the **worked numeric example** (compute $p_t$, $\hat\ell_t$, $\hat y_t$; compare with naive).
4. **Pure exploration regret** $=\frac Td\sum\Delta_i$.
5. **Greedy counterexample:** probability $0.16$, regret $\ge0.032(T-2)$.

### 13.3 Common traps
* **Policy class size:** $d^K$, not $d\cdot K$. Constant-arm policies are only $d$ of them.
* **Regret comparator:** CMAB regret is vs the best **policy** (per context), *not* the best single arm; the best policy's loss is $\le$ the best arm's loss.
* **$T_f$ vs $T$:** each EXP3 instance runs for $T_f$ rounds; do not plug $T$ into each instance's bound. The sum $\sum_f\sqrt{T_f}\le\sqrt{KT}$ is where $K$ enters.
* **Cauchy–Schwarz direction:** $\langle\mathbf 1,v\rangle\le\lVert\mathbf 1\rVert\lVert v\rVert$, with $\lVert\mathbf 1\rVert=\sqrt K$ (number of contexts), $\lVert v\rVert=\sqrt{\sum T_f}=\sqrt T$.
* **Experts give distributions** here (not 0/1 advice as in Week 1). Their loss is $\langle a_{t,l},\ell_t\rangle$.
* **Naive vs mixing:** naive draws an expert **first** then an arm; mixing draws an **arm** from the blended distribution. In the mixing version every expert gets a non-zero fake loss $\propto a_{t,l}(I_t)$; in the naive version only the drawn expert does.
* **Two layers of importance weighting:** arms ($/p_{t,i}$) and then a *linear map* to experts (no further division by $q$ in the mixing algorithm). Dividing by $q_{t,l}$ belongs to the naive algorithm only.
* **Distribution bookkeeping:** $q_t\in\Delta_L$ lives over **experts**; $p_t\in\Delta_d$ over **arms**; $a_{t,l}\in\Delta_d$ is expert $l$'s advice.
* **Regret decomposition** for stochastic bandits uses *expected pulls* $\mathbb E[N_i(T)]$; pure exploration has $\mathbb E N_i=T/d$ exactly.
* **A single counterexample suffices** to rule out an algorithm; a good algorithm must work on all instances.
* **Linear regret from a bad event of constant probability:** probability $0.16$ times linear regret $=$ linear expected regret.
* **"Greedy plays each arm once"** (not zero times); the first $d$ rounds are forced.

### 13.4 Quick numeric practice patterns
* Given $K=10$, $d=5$, $T=10^4$: CMAB-EXP3 bound $\sqrt{2\cdot10^4\cdot10\cdot5\ln5}\approx\sqrt{1.61\times10^6}\approx1269$; plain EXP3 bound $\sqrt{2\cdot10^4\cdot5\ln5}\approx401$ (but only against the weaker best-single-arm comparator).
* Mixing example: recompute $p_t=q_t^\top A$ (rows $=$ expert distributions) and $\hat y_t=A\hat\ell_t$.
* Pure exploration with $d=4$, gaps $(0,0.1,0.2,0.3)$, $T=1000$: $R=\frac{1000}{4}(0.6)=150$.
* Greedy with loss means $0.3$ (best) vs $0.7$: the bad event is "arm 1 returns $1$ (prob. $0.3$) and arm 2 returns $0$ (prob. $0.3$)", so its probability is $0.09$ and the regret is $\ge0.09\cdot0.4\,(T-2)=0.036\,(T-2)$.

---

## 14. Self-test questions with answers

**Q1.** Why is regret against the best fixed arm the wrong yardstick for contextual bandits?
**A.** Different contexts prefer different arms, so no single arm is good for all; it is a weak comparator. We compare with the best policy.

**Q2.** Define a policy and the number of policies.
**A.** A map $\pi:\mathcal F\to[d]$; $d^{\lvert\mathcal F\rvert}$ policies (including the $d$ constant ones).

**Q3.** Give the CMAB regret.
**A.** $R_T=\mathbb E\sum_t\ell_{t,I_t}-\min_\pi\sum_t\ell_{t,\pi(f_t)}$.

**Q4.** Describe CMAB-EXP3.
**A.** One EXP3 per context; on seeing $f_t$ invoke its instance, play its distribution, feed back the loss to that instance only.

**Q5.** Derive CMAB-EXP3's regret bound.
**A.** Split by context; each instance has regret $\le\sqrt{2T_fd\ln d}$; sum $=\sqrt{2d\ln d}\sum_f\sqrt{T_f}\le\sqrt{2d\ln d}\sqrt{KT}$ (Cauchy–Schwarz with $\sum T_f=T$).

**Q6.** When is CMAB-EXP3 poor, and why?
**A.** When the number of contexts $K$ is huge (maintain $K$ instances; $\sqrt K$ in the bound), because contexts are treated as unrelated and nothing is shared; structure in how losses depend on features would help.

**Q7.** What does an expert output in "bandits with expert advice", and what is its expected loss?
**A.** A distribution $a_{t,l}\in\Delta_d$ over arms; expected loss $\langle a_{t,l},\ell_t\rangle$.

**Q8.** Why can't we treat the expert-advice problem as a finite-context bandit?
**A.** The advice are points of the simplex (continuous); binning $L$ such points gives too many bins.

**Q9.** Describe the naive algorithm and its weakness.
**A.** Meta-EXP3 over experts: draw an expert from $q_t$, an arm from its advice, give the drawn expert fake loss $\ell/q_{t,e}$, others zero. Only one expert learns per round.

**Q10.** Describe the mixing algorithm.
**A.** $p_t=\sum_lq_{t,l}a_{t,l}$; draw $I_t\sim p_t$; $\hat\ell_{t,i}=\ell_{t,i}\mathbf 1[I_t=i]/p_{t,i}$; $\hat y_{t,l}=\langle a_{t,l},\hat\ell_t\rangle$; Hedge on $q$.

**Q11.** In the example with $a_1=(0.9,0.05,0.05)$, $a_2=(0.8,0.1,0.1)$, $a_3=(0.1,0.8,0.1)$, $q=(0.4,0.4,0.2)$, find $p_t$.
**A.** $(0.70,\,0.22,\,0.08)$.

**Q12.** In that example, arm 1 is drawn with loss $0.1$. What is the fake expert loss?
**A.** $\hat\ell=(0.1/0.7,0,0)$, $\hat y=(0.9,0.8,0.1)\cdot0.143=(0.129,0.114,0.014)$.

**Q13.** Why is the mixing estimator unbiased for the experts' expected loss?
**A.** $\mathbb E[\hat\ell_t\mid p_t]=\ell_t$ and $\hat y_{t,l}=\langle a_{t,l},\hat\ell_t\rangle$ is linear in $\hat\ell_t$, so $\mathbb E\hat y_{t,l}=\langle a_{t,l},\ell_t\rangle$.

**Q14.** State the stochastic-bandit regret decomposition.
**A.** $R_T=\sum_i\Delta_i\mathbb E[N_i(T)]$.

**Q15.** Regret of uniform-random play?
**A.** $R_T=\frac Td\sum_i\Delta_i$, linear whenever some arm is suboptimal.

**Q16.** Show pure greedy has linear regret.
**A.** Two Bernoulli arms with means $0.4,0.6$: with probability $0.4\cdot0.4=0.16$ the first pulls give arm 1 loss $1$, arm 2 loss $0$, so greedy commits to the bad arm for $T-2$ rounds, regret $\ge0.16\cdot0.2\cdot(T-2)$.

**Q17.** What does the lecture conclude about exploration and exploitation?
**A.** Both pure strategies give linear regret; algorithms must balance them (more than one pull per arm before committing, adaptive sampling).

---

## 15. Lecture slips / things to double-check

1. **Repeated lecture.** The middle of the file repeats the *Randomized Weighted Majority* lecture (RWM definition, potential $\Phi_t=\Phi_{t-1}(1-\epsilon\lvert p_t-y_t\rvert)$, finite vs infinite horizon). It is already in the Weeks 1–4 notes (§9–§10), so I did not repeat it.
2. **Names / notation in the transcript.** "CMAP/Cab" = CMAB (contextual multi-armed bandits); "lawn d" = $\ln d$; "$\text{ea}$" = $\eta$; the "fake loss vector" = importance-weighted loss estimate. The lecture does **not** name the mixing algorithm (it is EXP4-type).
3. **Constant in CMAB-EXP3 bound.** The lecture writes $\sqrt{T\lvert\mathcal F\rvert d\ln d}$ (O-notation). With EXP3's bound $\sqrt{2T_fd\ln d}$ the explicit bound is $\sqrt{2TKd\ln d}$. Also each instance's tuned $\eta_f$ needs $T_f$.
4. **Example numbers in the mixing algorithm are garbled** in the transcript ("$0.9\cdot4+8\cdot4+0.1\cdot2$", "$7,22,98$", "$.1\text{ by }.7$"). I reconstructed them as $a_1=(0.9,0.05,0.05)$, $a_2=(0.8,0.1,0.1)$, $a_3=(0.1,0.8,0.1)$, $q=(0.4,0.4,0.2)$, giving $p=(0.7,0.22,0.08)$, which sums to 1 and reproduces the lecture's $0.7$ and $0.22$. The exact entries of the experts' other coordinates were inferred from those totals.
5. **Naive algorithm:** the lecture calls it "EXP3 naive"; the importance weight there is $q_{t,e_t}$ (the probability of drawing the *expert*), not $p_{t,I_t}$ (probability of the arm).
6. **Regret of the mixing algorithm is not stated** in this lecture. The $\sqrt{2Td\ln L}$ bound in §7.5 is my own derivation (it matches the standard EXP4 result), and the naive bound $\sqrt{2TL\ln L}$ is the EXP3 bound with $L$ arms. Both are sanity-checked in the code; confirm against next week's analysis.
7. **Greedy example arithmetic.** The lecture computes $0.16\times0.2\,(T-1)$ and later says "$T-2$"; the correct count is $T-2$ (two forced pulls). Order-wise it does not matter. The lecture also describes arm outputs as "reward or loss" interchangeably; here I use **loss** (smaller $\mu$ is better), consistent with the rest of the course.
8. **Tie-breaking.** The lower bound uses only the bad event $(1,0)$; ties are broken towards arm 1 in the lecture's accounting, and that only matters for the exact value, not for the linear lower bound.
9. **"We will assume $L>d$."** Said as the interesting regime; the algorithm and bounds do not need it.
10. **Slight inconsistency in earlier wording:** "pure exploration" in the lecture means *uniform* random arm each round (no use of observed losses); do not confuse with EXP3's implicit exploration.

---

## 16. Runnable verification code

Checks numerically: the best-policy decomposition (brute force over all $d^K$ policies), CMAB-EXP3's regret vs $\sqrt{2TKd\ln d}$ (and that a single context-blind EXP3 does much worse against the best policy), the Cauchy–Schwarz step, the lecture's expert-advice example ($p_t=(0.7,0.22,0.08)$, both fake-loss vectors, exact unbiasedness of both estimators), the regret of the mixing vs naive algorithms (beyond the lecture), the exact regret of pure exploration, and the greedy counterexample ($0.16$, $0.032(T-2)$). Requires `numpy`. Save as `verify_w8.py` and run `python verify_w8.py`.

```python
"""Runnable sanity checks for SDM Week 8: contextual bandits (per-context EXP3), bandits with expert advice,
and why pure exploration / pure greedy fail for stochastic bandits."""
import itertools
import numpy as np
rng = np.random.default_rng(4)

def exp3_run(Lmat, eta, rng):
    """Plain EXP3 on a fixed T x d loss matrix; returns total loss suffered (realised)."""
    T, d = Lmat.shape; logw = np.zeros(d); total = 0.0
    for t in range(T):
        p = np.exp(logw - logw.max()); p /= p.sum()
        i = rng.choice(d, p=p); loss = Lmat[t, i]; total += loss
        logw[i] -= eta * loss / p[i]
    return total

# ---------- 1. Policy comparator decomposes per context:  min_pi sum_t l_{t,pi(f_t)} = sum_f min_j sum_{t: f_t=f} l_{t,j} ----------
K, d, T = 3, 2, 12
ctx = rng.integers(0, K, T); Lm = rng.random((T, d))
brute = min(sum(Lm[t, pi[ctx[t]]] for t in range(T)) for pi in itertools.product(range(d), repeat=K))   # all d^K policies
per_ctx = sum(min(Lm[ctx == f, j].sum() for j in range(d)) for f in range(K))
print(f"[policy class] #policies = d^K = {d**K}; brute-force best policy = {brute:.4f}; sum of per-context bests = {per_ctx:.4f}")
assert np.isclose(brute, per_ctx)

# ---------- 2. CMAB-EXP3: one EXP3 per context. Regret <= sqrt(2 T K d ln d)  (Cauchy-Schwarz over contexts) ----------
K, d, T = 8, 6, 12000
ctx = rng.integers(0, K, T)
Lmat = rng.random((T, d))
best_arm = rng.integers(0, d, K)                          # each context has its own best arm
for t in range(T): Lmat[t, best_arm[ctx[t]]] = max(Lmat[t, best_arm[ctx[t]]] - 0.25, 0.0)
Tf = np.bincount(ctx, minlength=K)
assert np.sum(np.sqrt(Tf)) <= np.sqrt(K * T) + 1e-9       # Cauchy-Schwarz: sum_f sqrt(T_f) <= sqrt(K * sum_f T_f)
regs = []
for _ in range(20):
    alg = sum(exp3_run(Lmat[ctx == f], np.sqrt(2 * np.log(d) / (Tf[f] * d)), rng) for f in range(K))
    best_policy = sum(Lmat[ctx == f].sum(0).min() for f in range(K))
    regs.append(alg - best_policy)
# a single EXP3 that ignores the context is compared against the same best *policy*
single = np.mean([exp3_run(Lmat, np.sqrt(2 * np.log(d) / (T * d)), rng) for _ in range(20)]) - sum(Lmat[ctx == f].sum(0).min() for f in range(K))
print(f"[CMAB-EXP3] mean regret vs best policy = {np.mean(regs):.1f} <= sqrt(2 T K d ln d) = {np.sqrt(2*T*K*d*np.log(d)):.1f}")
print(f"[plain EXP3, context ignored] regret vs best policy = {single:.1f}  (much larger: one arm can't be good for every context)")
assert np.mean(regs) <= np.sqrt(2 * T * K * d * np.log(d)) and single > 1.5 * np.mean(regs)

# ---------- 3. Bandits with expert advice: lecture's worked example ----------
A = np.array([[0.9, 0.05, 0.05], [0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])   # rows = experts' distributions over 3 arms
q = np.array([0.4, 0.4, 0.2])                                          # belief over experts
p = q @ A                                                              # mixed distribution over arms
print("[BEA example] p_t = q^T A =", p.round(3), "(lecture: 0.7, 0.22, 0.08)")
assert np.allclose(p, [0.7, 0.22, 0.08])
l_obs = 0.1                                                            # arm 1 drawn, loss 0.1
lhat = np.array([l_obs / p[0], 0, 0]); y_mix = A @ lhat               # EXP4-style fake expert loss
y_naive = np.array([l_obs / q[0], 0, 0])                               # naive: expert 1 was drawn first
print("             fake expert loss (mixing)   =", y_mix.round(4)); print("             fake expert loss (naive)    =", y_naive.round(4))

# exact unbiasedness of both estimators: E[y_l] = <a_l, ell>
ell = np.array([0.1, 0.5, 0.8]); target = A @ ell
E_mix = sum(p[i] * (A @ np.where(np.arange(3) == i, ell[i] / p[i], 0.0)) for i in range(3))
E_naive = np.zeros(3)
for e in range(3):                                                     # expert e drawn w.p. q[e]; then arm i ~ A[e]
    for i in range(3):
        y = np.zeros(3); y[e] = ell[i] / q[e]; E_naive += q[e] * A[e, i] * y
print("             target <a_l, ell> =", target.round(4), "| E[mixing] =", E_mix.round(4), "| E[naive] =", E_naive.round(4))
assert np.allclose(E_mix, target) and np.allclose(E_naive, target)

# ---------- 4. (beyond lecture) regret of the mixing algorithm vs the naive one:  sqrt(2 T d ln L)  vs  sqrt(2 T L ln L) ----------
Lx, d, T = 60, 5, 4000
Lmat = rng.random((T, d)); Aadv = rng.dirichlet(np.ones(d) * 0.3, size=(T, Lx))        # T x L x d expert advice
good = np.argmin(Lmat, axis=1); Aadv[:, 0, :] = 0.05 / (d - 1); Aadv[np.arange(T), 0, good] = 0.95   # expert 0 follows the best arm
exp_loss = np.einsum('tld,td->tl', Aadv, Lmat)                                          # <a_{t,l}, ell_t>
def run(mix, eta, rng):
    logq = np.zeros(Lx); total = 0.0
    for t in range(T):
        qt = np.exp(logq - logq.max()); qt /= qt.sum()
        if mix:
            pt = qt @ Aadv[t]; i = rng.choice(d, p=pt); loss = Lmat[t, i]; total += loss
            lh = np.zeros(d); lh[i] = loss / pt[i]; y = Aadv[t] @ lh                    # every expert is updated
        else:
            e = rng.choice(Lx, p=qt); i = rng.choice(d, p=Aadv[t, e]); loss = Lmat[t, i]; total += loss
            y = np.zeros(Lx); y[e] = loss / qt[e]                                       # only the drawn expert is updated
        logq -= eta * y
    return total - exp_loss.sum(0).min()
r_mix = np.mean([run(True, np.sqrt(2 * np.log(Lx) / (T * d)), rng) for _ in range(10)])
r_naive = np.mean([run(False, np.sqrt(2 * np.log(Lx) / (T * Lx)), rng) for _ in range(10)])
print(f"[BEA, L={Lx}, d={d}] mixing: {r_mix:.1f} <= sqrt(2 T d ln L) = {np.sqrt(2*T*d*np.log(Lx)):.1f};  naive: {r_naive:.1f} <= sqrt(2 T L ln L) = {np.sqrt(2*T*Lx*np.log(Lx)):.1f}")
assert r_mix <= np.sqrt(2 * T * d * np.log(Lx)) and r_naive <= np.sqrt(2 * T * Lx * np.log(Lx)) and r_mix < r_naive

# ---------- 5. Stochastic bandits: pure exploration has regret exactly (T/d) sum_i Delta_i ----------
mu = np.array([0.4, 0.6]); delta = mu - mu.min(); T = 2000
reg = np.mean([np.sum(delta[rng.integers(0, 2, T)]) for _ in range(2000)])
print(f"[pure exploration] simulated regret = {reg:.1f}; formula (T/d) sum Delta_i = {T/2*delta.sum():.1f}")
assert abs(reg - T / 2 * delta.sum()) < 0.03 * T

# ---------- 6. Pure greedy (pull each arm once, commit to the smaller empirical loss, ties -> arm 1): linear regret ----------
def greedy_pseudo_regret(T, rng):
    x1, x2 = rng.random() < mu[0], rng.random() < mu[1]                  # Bernoulli losses of the two initial pulls
    commit = 1 if x2 < x1 else 0                                         # arm 2 only if it strictly looks better
    return delta[0] * 1 + delta[1] * 1 + delta[commit] * (T - 2), commit == 1
runs = [greedy_pseudo_regret(T, rng) for _ in range(200000)]
bad = np.mean([b for _, b in runs]); mean_reg = np.mean([r for r, _ in runs])
exact = 0.16 * 0.2 * (T - 2) + 0.2                                       # P(bad)=0.4*0.4, gap 0.2, plus the forced pull of arm 2
print(f"[pure greedy] P(commit to bad arm) = {bad:.3f} (exact 0.16); E[regret] = {mean_reg:.2f} vs exact {exact:.2f}; "
      f"E[regret]/T = {mean_reg/T:.4f} -> {0.032}")
assert abs(bad - 0.16) < 0.005 and abs(mean_reg - exact) < 0.02 * exact
for TT in (10**3, 10**4, 10**5):
    print(f"             T = {TT:>6}: exact E[regret]/T = {(0.032*(TT-2)+0.2)/TT:.4f}  (does not go to 0)")
print("All checks passed.")

```

**Expected output** (the random-simulation numbers vary slightly with the seed/NumPy version; the assertions are what matter)

```text
[policy class] #policies = d^K = 8; brute-force best policy = 5.9481; sum of per-context bests = 5.9481
[CMAB-EXP3] mean regret vs best policy = 847.4 <= sqrt(2 T K d ln d) = 1436.7
[plain EXP3, context ignored] regret vs best policy = 1915.8  (much larger: one arm can't be good for every context)
[BEA example] p_t = q^T A = [0.7  0.22 0.08] (lecture: 0.7, 0.22, 0.08)
             fake expert loss (mixing)   = [0.1286 0.1143 0.0143]
             fake expert loss (naive)    = [0.25 0.   0.  ]
             target <a_l, ell> = [0.155 0.21  0.49 ] | E[mixing] = [0.155 0.21  0.49 ] | E[naive] = [0.155 0.21  0.49 ]
[BEA, L=60, d=5] mixing: 208.8 <= sqrt(2 T d ln L) = 404.7;  naive: 841.8 <= sqrt(2 T L ln L) = 1401.9
[pure exploration] simulated regret = 200.0; formula (T/d) sum Delta_i = 200.0
[pure greedy] P(commit to bad arm) = 0.161 (exact 0.16); E[regret] = 64.41 vs exact 64.14; E[regret]/T = 0.0322 -> 0.032
             T =   1000: exact E[regret]/T = 0.0321  (does not go to 0)
             T =  10000: exact E[regret]/T = 0.0320  (does not go to 0)
             T = 100000: exact E[regret]/T = 0.0320  (does not go to 0)
All checks passed.
```

---
*End of Week 8 notes.*
