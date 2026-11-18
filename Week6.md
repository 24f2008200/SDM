# Sequential Decision Making — Week 6 Notes
**Follow the Regularised Leader (FTRL), Online Gradient Descent, Entropic Regularisation = Hedge, Online Convex Optimisation, and SGD**

> Continuation of the Week 1–4 and Week 5 notes. Built from lecture transcript L30–38 (the end of Part 1: online learning with full information).
> Math is GitHub-flavoured LaTeX (`$...$`, `$$...$$`).
> Notation: $S$ = action set, $w_t\in S$ = learner's play, $f_t$ = loss function, $z_t$ = loss vector of a linear loss $f_t(w)=\langle w,z_t\rangle$, $\eta$ = regularisation/step-size parameter (written "ea" in the transcript), $\lVert\cdot\rVert$ = Euclidean norm.

---

## Table of Contents
1. [One-page summary](#1-one-page-summary)
2. [FTRL: definition and the "round zero" trick](#2-ftrl-definition-and-the-round-zero-trick)
3. [Generic FTRL regret bound](#3-generic-ftrl-regret-bound)
4. [FTRL with $L_2$ regulariser on linear losses = online gradient descent](#4-ftrl-with-l_2-regulariser-on-linear-losses--online-gradient-descent)
5. [Regret of FTRL with $L_2$ regulariser: $O(BL\sqrt T)$](#5-regret-of-ftrl-with-l_2-regulariser-oblsqrt-t)
6. [Is $L_2$-FTRL a replacement for Hedge? The geometry problem](#6-is-l_2-ftrl-a-replacement-for-hedge-the-geometry-problem)
7. [Entropic regulariser: FTRL = Hedge](#7-entropic-regulariser-ftrl--hedge)
8. [Table of regimes: action space, adversary, regulariser, regret](#8-table-of-regimes)
9. [Online convex optimisation (OCO)](#9-online-convex-optimisation-oco)
10. [Linearisation and online (sub)gradient descent](#10-linearisation-and-online-subgradient-descent)
11. [Curvature: what convexity alone cannot give](#11-curvature-what-convexity-alone-cannot-give)
12. [Application: stochastic gradient descent (SGD)](#12-application-stochastic-gradient-descent-sgd)
13. [Where next: Part 2 (partial information)](#13-where-next-part-2-partial-information)
14. [Formula table](#14-formula-table)
15. [Quiz preparation tips](#15-quiz-preparation-tips)
16. [Self-test questions with answers](#16-self-test-questions-with-answers)
17. [Lecture slips / things to double-check](#17-lecture-slips--things-to-double-check)
18. [Runnable verification code](#18-runnable-verification-code)

---

## 1. One-page summary

| Idea | Statement |
|---|---|
| **FTRL** | $w_t=\arg\min_{w\in S}\Big[\sum_{i=1}^{t-1}f_i(w)+R(w)\Big]$ — FTL plus a regulariser $R$ to stop the leader from jumping |
| **Round-zero trick** | FTRL $=$ FTL on the sequence $f_0=R,f_1,f_2,\dots$ where the *learner* (not the adversary) chooses $f_0$ |
| **Generic bound** | $R_T(u)\le R(u)-R(w_1)+\sum_{t=1}^T[f_t(w_t)-f_t(w_{t+1})]$ |
| **$L_2$ regulariser, linear loss** | $R(w)=\frac1{2\eta}\lVert w\rVert^2\ \Rightarrow\ w_{t+1}=w_t-\eta z_t$ (**online gradient descent**) |
| **Regret ($L_2$)** | $\le\frac{\lVert u\rVert^2}{2\eta}+\eta\sum_t\lVert z_t\rVert^2\le\frac{B^2}{2\eta}+\eta L^2T$; with $\eta^*=\frac{B}{L\sqrt{2T}}$: $\ \sqrt2\,BL\sqrt T=O(\sqrt T)$ |
| **Hedge setting with $L_2$** | simplex $\subset$ unit ball ($B=1$), hypercube $[-1,1]^d$ needs $L=\sqrt d$ $\Rightarrow$ $O(\sqrt{Td})$, **exponentially worse in $d$** than Hedge's $O(\sqrt{T\ln d})$ |
| **Why** | The regulariser's geometry ($L_2$ balls) does not match the simplex; probabilities should be compared by **entropy** |
| **Entropic regulariser** | $R(w)=\frac1\eta\sum_iw_i\ln w_i$ on $\Delta_d$ $\Rightarrow$ $w_{t+1,j}=\dfrac{e^{-\eta L_{t,j}}}{\sum_ie^{-\eta L_{t,i}}}$ $=$ **Hedge** exactly |
| **OCO** | $f_t$ convex. Regret $\le$ regret of the linear game with $z_t\in\partial f_t(w_t)$ |
| **Online (sub)gradient descent** | $w_{t+1}=w_t-\eta z_t$, $z_t\in\partial f_t(w_t)$; regret $O(BL\sqrt T)$ for any convex $L$-Lipschitz losses |
| **Strongly convex losses** | regret improves to $O(\log T)$ (stated, not proved) |
| **SGD** | SGD $=$ online gradient descent with noisy gradients and the same $f$ every round; $\mathbb E[f(\bar x_T)]-f(x^*)=O(1/\sqrt T)$ |

**Five ideas to remember**
1. **Regularisation = stability.** One extra quadratic term at "round zero" makes each step a small gradient step instead of an endpoint flip.
2. **FTRL with $L_2$ is gradient descent**; **FTRL with negative entropy is Hedge** (multiplicative weights). The regulariser fixes the algorithm.
3. **Match the regulariser to the geometry of the action space.** Entropy for the simplex, $L_2$ for balls.
4. **Convex $\Rightarrow$ linearise:** the tangent lower-bounds a convex function, so linear-game regret upper-bounds convex-game regret. Online gradient descent solves OCO.
5. **Regret bounds imply optimisation guarantees:** averaging the iterates of an online algorithm (Jensen) turns $O(\sqrt T)$ regret into $O(1/\sqrt T)$ suboptimality, which is why SGD works.

---

## 2. FTRL: definition and the "round zero" trick

**Recap (Week 5).** FTL, $w_t=\arg\min_w\sum_{i<t}f_i(w)$, has $O(\log T)$ regret for quadratic losses but $\Theta(T)$ regret for linear losses because it jumps between extreme points.

**Fix.** Control model complexity the standard ML way: **regularise**.

$$
\boxed{\ w_t^{\text{FTRL}}=\arg\min_{w\in S}\Big[\sum_{i=1}^{t-1}f_i(w)+R(w)\Big]\ }
$$

* $R(w)$ is the regulariser. A typical choice is a norm: it penalises large models, so adding one more function cannot move $w$ far.

### 2.1 FTRL is FTL with an extra first round
Imagine a **round zero** in which the adversary is forced to reveal $f_0=R$:

$$
f_0=R,\ f_1,\ f_2,\ \dots,\ f_T .
$$

FTL on this sequence picks $w_t=\arg\min_w\sum_{i=0}^{t-1}f_i(w)=\arg\min_w\big[R(w)+\sum_{i=1}^{t-1}f_i(w)\big]$, which is exactly FTRL.

The difference from Week 5: **we control $f_0$** (we choose what the regulariser is); the adversary only controls $f_1,\dots,f_T$. With a quadratic $f_0$, FTL's good behaviour on quadratics can leak into the whole game.

---

## 3. Generic FTRL regret bound

Apply the FTL result of Week 5 (regret $\le$ FTL–BTL gap) to the sequence $f_0,f_1,\dots,f_T$:

$$
\sum_{t=0}^Tf_t(w_t)-\sum_{t=0}^Tf_t(u)\ \le\ \sum_{t=0}^T\big[f_t(w_t)-f_t(w_{t+1})\big].
$$

Separate the $t=0$ terms ($f_0=R$) from the real game ($t\ge1$). The term $f_0(w_0)$ cancels on both sides and $f_0(u)=R(u)$, $f_0(w_1)=R(w_1)$. Rearranging:

$$
\boxed{\ \underbrace{\sum_{t=1}^Tf_t(w_t)-\sum_{t=1}^Tf_t(u)}_{\text{regret of FTRL vs }u}\ \le\ R(u)-R(w_1)+\sum_{t=1}^T\big[f_t(w_t)-f_t(w_{t+1})\big]\ }
$$

**Choosing the first point.** In round 1 FTRL has already "seen" $R$, so $w_1=\arg\min_wR(w)$. Regularisers are typically non-negative norms with minimum $0$, so we can take $R(w_1)=0$:

$$
R_T(u)\le R(u)+\sum_{t=1}^T\big[f_t(w_t)-f_t(w_{t+1})\big].\tag{A}
$$

Meaning: **regret = (price of the regulariser at the comparator) + (sensitivity of the algorithm to one extra function)**. Regularisation trades a penalty $R(u)$ for stability.

---

## 4. FTRL with $L_2$ regulariser on linear losses = online gradient descent

### 4.1 Setting
* Linear losses: $f_t(w)=\langle w,z_t\rangle$ for $t\ge1$ (the adversary supplies $z_t$).
* Regulariser: $R(w)=\dfrac1{2\eta}\lVert w\rVert^2$ ($\eta>0$ is the "balancing" constant that would be cross-validated in ML; the $\frac12$ cancels the 2 from differentiating $\lVert w\rVert^2$).
* Action space $S=\mathbb R^d$ for now (see §5.3 for the ball).

### 4.2 Closed form
$$
w_t=\arg\min_{w\in\mathbb R^d}\Big[\Big\langle w,\sum_{i<t}z_i\Big\rangle+\frac1{2\eta}\lVert w\rVert^2\Big].
$$
Set the gradient to zero: $\sum_{i<t}z_i+\frac1\eta w=0$, so

$$
\boxed{\ w_t=-\eta\sum_{i=1}^{t-1}z_i\ }
\qquad\Longrightarrow\qquad
\boxed{\ w_{t+1}=w_t-\eta\,z_t\ }
$$

(Also $w_1=0=\arg\min R$.)

### 4.3 Interpretation: online gradient descent
Since $\nabla f_t(w)=z_t$ everywhere (linear loss), the update is "move along the **negative gradient** with step size $\eta$": **online gradient descent (OGD)**.

* Each step is a *small* move of size $\eta\lVert z_t\rVert$, however the adversary chooses $z_t$. This removes the endpoint-flipping that broke FTL.
* So FTRL with $L_2$ regularisation $\equiv$ OGD. (With $f_t$ not linear, OGD uses the gradient/subgradient at $w_t$; see §10.)

---

## 5. Regret of FTRL with $L_2$ regulariser: $O(BL\sqrt T)$

### 5.1 Sensitivity term
From $w_{t+1}=w_t-\eta z_t$:
$$
f_t(w_t)-f_t(w_{t+1})=\langle w_t-w_{t+1},z_t\rangle=\eta\,\lVert z_t\rVert^2 .
$$

### 5.2 Plug into bound (A)
$$
R_T(u)\le\frac{\lVert u\rVert^2}{2\eta}+\eta\sum_{t=1}^T\lVert z_t\rVert^2 .
$$
Assumptions:
* **(A)** the adversary's loss vectors are bounded: $\lVert z_t\rVert\le L$;
* **(B)** the **comparator class** is bounded: $\lVert u\rVert\le B$.

$$
\boxed{\ R_T\le\frac{B^2}{2\eta}+\eta L^2T\ }
$$

**Optimising $\eta$:** $\dfrac{d}{d\eta}=-\dfrac{B^2}{2\eta^2}+L^2T=0\Rightarrow\eta^2=\dfrac{B^2}{2L^2T}$:

$$
\boxed{\ \eta^*=\frac{B}{L\sqrt{2T}}\ },\qquad
\frac{B^2}{2\eta^*}=\frac{BL\sqrt{2T}}{2}=\frac{BL\sqrt T}{\sqrt2},\quad
\eta^*L^2T=\frac{BL\sqrt T}{\sqrt2}
$$

$$
\boxed{\ R_T\le\sqrt2\,BL\sqrt T=O(BL\sqrt T)\ }
$$

(The two terms are equal at the optimum, as for Hedge.) **Sublinear regret for linear losses**: adding just *one* quadratic function at the start is enough. It is not $O(\log T)$, because the adversary is still linear.

$\eta^*$ depends on $B$, $L$ and $T$. For an unknown horizon use the **doubling trick**, as before.

### 5.3 Remarks the lecture made
* **Improper learning.** The learner $w_t=-\eta\sum z_i$ may have norm larger than $B$; only the *comparator* is restricted to $\lVert u\rVert\le B$. When the learner plays from a larger class than the comparator, it is called **improper learning**. (Consequently measured regret can even be negative, see the code.)
* **Is the restriction the reason it works? No.** Forcing the learner into the ball $\lVert w\rVert\le B$ gives **projected online gradient descent**: take the gradient step, then if $\lVert w\rVert>B$ rescale $w\leftarrow B\,w/\lVert w\rVert$ (the closest point of the ball lies in the same direction). Its regret is again $O(BL\sqrt T)$ (the lecture states this without proof; the analysis needs convex-conjugate tools).

---

## 6. Is $L_2$-FTRL a replacement for Hedge? The geometry problem

**Hedge's game:** play $p_t\in\Delta_d$, receive loss $\ell_t\in[-1,1]^d$ (hypercube), suffer $\langle p_t,\ell_t\rangle$. Hedge regret: $O(\sqrt{T\ln d})$.

**Force it into the FTRL-$L_2$ framework** (find $B$ and $L$):

* **Choosing $L$:** the adversary's vectors must lie in an $L$-ball. The largest norm in $[-1,1]^d$ is the all-ones vector: $\lVert\mathbf 1\rVert=\sqrt d$. So $L=\sqrt d$.
* **Choosing $B$:** the comparators are points of the simplex. Corners $e_i$ have norm $1$; the centre $(\frac1d,\dots,\frac1d)$ has norm $\sqrt{d\cdot\frac1{d^2}}=\frac1{\sqrt d}$; norm grows from the centre to the corners. So the unit ball contains the simplex: $B=1$.

$$
R_T(L_2\text{-FTRL})\le\sqrt2\cdot1\cdot\sqrt d\cdot\sqrt T=O(\sqrt{Td})\quad\text{vs}\quad R_T(\text{Hedge})=O(\sqrt{T\ln d}).
$$

Same $\sqrt T$, but the dependence on $d$ is **$\sqrt d$ vs $\sqrt{\ln d}$**: an exponential gap. For many experts (huge $d$), Hedge is far better.

| $d$ | $\sqrt{2dT}$ ($L_2$-FTRL) | $\sqrt{2T\ln d}$ (Hedge) |
|---|---|---|
| 10 | 447 | 215 |
| 100 | 1414 | 303 |
| 1000 | 4472 | 372 |
| 10000 | 14142 | 429 |

(for $T=10{,}000$)

### 6.1 Why does $L_2$ fail here? Geometry
* A regulariser tells the algorithm which points to treat as "equally complex": its **contours** (level sets).
* $L_2$ contours on the simplex are **circles** centred at $(\frac1d,\dots,\frac1d)$. But points of the simplex are **probability distributions**; two distributions with the same $L_2$ length can be very different, and the $L_2$ norm is not the natural way to compare them.
* For probability distributions the natural "size" is **entropy**: it measures uncertainty/surprise.
  * $(1,0,0)$: no surprise (always item 1), entropy $0$.
  * Uniform $(\frac13,\frac13,\frac13)$: maximal uncertainty, maximal entropy.
* Entropy contours on the simplex are *not* circles: they bulge towards the edges. The regulariser should match the geometry of the action space.

---

## 7. Entropic regulariser: FTRL $=$ Hedge

### 7.1 The regulariser
Use the **negative entropy** $-H(w)=\sum_jw_j\ln w_j$. It is convex, and since we *minimise*, it favours high-entropy (more uncertain, more spread-out) distributions:

$$
R(w)=\frac1\eta\sum_{j=1}^dw_j\ln w_j,\qquad w\in\Delta_d .
$$

### 7.2 The FTRL problem (linear losses on the simplex)
Let $L_t=\sum_{i\le t}z_i$ be the cumulative loss vector (to avoid writing sums). In round $t+1$:

$$
w_{t+1}=\arg\min_{w\in\Delta_d}\ \Big[\langle w,L_t\rangle+\frac1\eta\sum_jw_j\ln w_j\Big],\quad
\Delta_d=\Big\{w:\ \sum_jw_j=1,\ w_j\ge0\Big\}.
$$

The objective is convex, so the **KKT conditions** are necessary and sufficient.

### 7.3 Solving with Lagrange multipliers
Constraints: one equality $\sum_jw_j-1=0$ (multiplier $\lambda$) and $d$ inequalities $-w_j\le0$ (multipliers $\beta_j\ge0$). Lagrangian:

$$
\mathcal L(w,\lambda,\beta)=\langle w,L_t\rangle+\frac1\eta\sum_jw_j\ln w_j+\lambda\Big(\sum_jw_j-1\Big)+\sum_j\beta_j(-w_j).
$$

**Stationarity** ($\partial\mathcal L/\partial w_j=0$; note $\frac{d}{dw}w\ln w=1+\ln w$):
$$
L_{t,j}+\frac1\eta\big(1+\ln w_j\big)+\lambda-\beta_j=0\qquad\forall j.\tag{1}
$$
**Complementary slackness:** $\lambda(\sum_jw_j-1)=0$ and $\beta_jw_j=0$ for all $j$.

For every $j$ with $w_j>0$, complementary slackness forces $\beta_j=0$, so (1) gives
$$
\ln w_j=-\eta\lambda-\eta L_{t,j}-1\ \Longrightarrow\ w_j=e^{-\eta L_{t,j}}\cdot e^{-\eta\lambda-1}.
$$
The factor $e^{-\eta\lambda-1}$ is the same for every $j$: it is a **normaliser**. Impose $\sum_jw_j=1$:

$$
\boxed{\ w_{t+1,j}=\frac{e^{-\eta L_{t,j}}}{\sum_{i=1}^de^{-\eta L_{t,i}}}\ }
$$

(Remark: since $w\ln w$ has infinite slope at $0$, every coordinate of the optimum is strictly positive, so the "$w_j>0$" case covers all $j$.)

### 7.4 Recursive form: the Hedge update
Since $L_{t,j}=L_{t-1,j}+z_{t,j}$:
$$
w_{t+1,j}=\frac{e^{-\eta L_{t-1,j}}\,e^{-\eta z_{t,j}}}{\sum_ie^{-\eta L_{t,i}}}
\ \propto\ w_{t,j}\,e^{-\eta z_{t,j}}.
$$
In Hedge terms: multiply each (unnormalised) weight by $e^{-\eta z_{t,j}}$ and **renormalise**. This is *exactly* the Hedge update ($w_{t+1,i}=w_{t,i}e^{-\eta\ell_{t,i}}$, $p=w/\Phi$).

> **Careful with the lecture's notation:** here the *normalised* probability vector is called $w_t$ (the FTRL iterate), whereas in Hedge $w$ denoted the *unnormalised* weights and $p=w/\Phi$ the probabilities. $\Phi_t=\sum_iw_{t,i}$ plays the role of the normaliser.

### 7.5 Conclusion
$$
\boxed{\ \text{FTRL with entropic regulariser on the simplex}\ =\ \text{Hedge}\ }
$$
so its regret is already known: $O(\sqrt{T\ln d})$. No new analysis needed.

**Why this matters**
* Hedge's $e^{-\eta\ell}$ rule looked "pulled out of a hat" (arising from weighted majority). Now it is *derived*: it is what you get from a principled regulariser matching the simplex.
* Recipe for new problems: **pick a regulariser fitting the geometry of the action space**, then FTRL gives an algorithm and a regret bound.
* Same destination from two roads: Majority $\to$ WM $\to$ RWM $\to$ Hedge (Weeks 1–4) vs FTL $\to$ FTRL ($L_2$: OGD) $\to$ FTRL (entropy: Hedge).

---

## 8. Table of regimes

| Action space | Adversary (loss) | Regularisation | Algorithm | Regret |
|---|---|---|---|---|
| $\mathbb R^d$ | **quadratic** | none | FTL | $O(\log T)$ |
| $\mathbb R^d$ (or ball) | **linear** | none | FTL | $\Theta(T)$ — fails |
| $\mathbb R^d$ (or ball $\lVert w\rVert\le B$) | **linear** ($\lVert z_t\rVert\le L$) | $L_2$ | FTRL = (projected) OGD | $O(BL\sqrt T)$ |
| simplex $\Delta_d$ | **linear** | $L_2$ | FTRL | $O(\sqrt{Td})$ |
| simplex $\Delta_d$ | **linear** | **entropy** | FTRL = **Hedge** | $O(\sqrt{T\ln d})$ |
| convex set | **convex** (Lipschitz) | $L_2$ via linearisation | online (sub)gradient descent | $O(BL\sqrt T)$ |
| convex set | **strongly convex** | (not covered) | (OGD variants) | $O(\log T)$ (stated only) |

---

## 9. Online convex optimisation (OCO)

### 9.1 Why
Most ML losses are **convex** (hinge, logistic, exponential/boosting, squared). Instead of analysing each separately, analyse them all at once.

### 9.2 Convex function (the definition used in the course)
$S$ must be a **convex set** (the segment between any two points of $S$ lies in $S$; e.g. an interval, $\mathbb R^d$, a ball). A function $f:S\to\mathbb R$ is **convex** if

$$
\forall w\in S\ \ \exists z\ \ \text{such that}\ \ \forall u\in S:\qquad f(u)\ \ge\ f(w)+\langle u-w,\ z\rangle .
$$

**Geometric meaning:** at every point $w$ there is a **tangent (supporting) line/hyperplane that lies entirely below the function**. A non-convex function has a point whose tangent rises above the graph somewhere.

* Such a $z$ is called a **subgradient** of $f$ at $w$; the set of all of them is $\partial f(w)$.
* **Differentiable** at $w$: the subgradient is **unique** and equals $\nabla f(w)$.
* **Non-differentiable** points (kinks) can still be convex: infinitely many supporting lines exist there (e.g. $f(x)=|x|$ at $0$: every slope in $[-1,1]$ works; the hinge loss at its kink).
* **Linear functions are convex** (the tangent equals the function). So linear losses are the "flattest" convex case, with no curvature.

### 9.3 OCO protocol
For $t=1,\dots,T$: play $w_t\in S$; receive a **convex** function $f_t:S\to\mathbb R$; suffer $f_t(w_t)$.

$$
R_T=\sum_{t=1}^Tf_t(w_t)-\min_{u\in S}\sum_{t=1}^Tf_t(u).
$$

Example: hinge loss $f_t(w)=\max\{0,\,1-y_t\,w^\top x_t\}$ gives an online hinge-loss problem.

---

## 10. Linearisation and online (sub)gradient descent

### 10.1 The reduction
Let $z_t\in\partial f_t(w_t)$ (the gradient if $f_t$ is differentiable). By convexity, for any $u$:
$$
f_t(u)\ \ge\ f_t(w_t)+\langle u-w_t,z_t\rangle
\ \Longrightarrow\
f_t(w_t)-f_t(u)\ \le\ \langle w_t-u,\ z_t\rangle .
$$
Sum over $t$:

$$
\boxed{\ R_T(u)=\sum_t\big[f_t(w_t)-f_t(u)\big]\ \le\ \underbrace{\sum_t\langle w_t,z_t\rangle-\sum_t\langle u,z_t\rangle}_{\text{regret of the linear game with losses }z_t}\ }
$$

**Interpretation.** Pretend the adversary handed us the **linear** function $w\mapsto\langle w,z_t\rangle$ (the tangent at $w_t$) instead of $f_t$. The tangent lies *below* $f_t$, and the two coincide at $w_t$ (so our loss is unchanged there). Our own loss is identical but the comparator's loss only goes down, so the linear regret is *at least* the convex regret.

### 10.2 The algorithm
Run FTRL ($L_2$) on the linear game with $z_t=\nabla f_t(w_t)$:

$$
w_{t+1}=w_t-\eta\,z_t,\qquad z_t\in\partial f_t(w_t),\quad w_1=0.
$$

* Differentiable $f_t$: **online gradient descent (OGD)**.
* Non-differentiable $f_t$: **online subgradient descent**.
* Note $z_t$ is *computed by the learner* at $w_t$; the adversary supplies the function.

### 10.3 Regret
Assume $\lVert z_t\rVert\le L$ for all $t$ (**gradients bounded**; this is a **Lipschitz** condition: the function cannot change too fast) and compare to $\lVert u\rVert\le B$. By §5:

$$
\boxed{\ R_T(\text{OGD})\le\sqrt2\,BL\sqrt T=O(BL\sqrt T)\quad\text{with }\eta^*=\frac{B}{L\sqrt{2T}}\ }
$$

**One algorithm, any convex loss sequence** (hinge, logistic, exponential, squared, ...), as long as gradients are bounded.

---

## 11. Curvature: what convexity alone cannot give

* We *threw away* curvature by linearising. Is that wasteful? Without further assumptions, **no**: linear functions are convex, so the adversary may play linear losses, for which $\sqrt T$ regret is the best possible. Any bound valid for *all* convex adversaries cannot beat $\sqrt T$.
* **With extra assumptions the rate improves.** A **strongly convex** function curves at least like a quadratic (with a minimum curvature parameter); a **strictly convex** function has no flat pieces. For strongly convex losses, regret $O(\log T)$ is achievable (the quadratic adversary of Week 5 is the extreme "just strongly convex" case). Stated in the lecture without proof.
* **Rule of thumb:** more assumptions on the adversary $\Rightarrow$ stronger bounds. Linear $\to\sqrt T$; strongly convex $\to\log T$.

---

## 12. Application: stochastic gradient descent (SGD)

Regret bounds also prove something about **offline optimisation**.

### 12.1 Setting
Minimise a convex $f$: $\ \min_xf(x)$. The gradient is unavailable (e.g. $f$ is an average over a billion data points); instead an **oracle** returns a **noisy gradient** $\tilde z$ with

$$
\mathbb E[\tilde z\mid x]=\nabla f(x).
$$

In practice: sample a (mini-)batch uniformly at random and compute its gradient; uniform sampling makes it unbiased.

### 12.2 Algorithm
$$
x_{t+1}=x_t-\eta\,\tilde z_t,\qquad \tilde z_t\ \text{noisy gradient at }x_t,\qquad
\text{output }\ \bar x_T=\frac1T\sum_{t=1}^Tx_t\ \ (\text{the }\textbf{average}\text{, not }x_T).
$$

### 12.3 Viewing SGD as OGD
SGD is online gradient descent where (i) the adversary plays the **same** $f$ every round, and (ii) the linear loss vector is the noisy gradient $\tilde z_t$. The linear-game regret bound holds for **any** sequence $\tilde z_t$ (it never used that they are gradients), so with $\lVert\tilde z_t\rVert\le L$ and $\lVert x^*\rVert\le B$:

$$
\sum_{t=1}^T\langle x_t-x^*,\tilde z_t\rangle\le\sqrt2\,BL\sqrt T\qquad\text{(for every realisation)}.
$$

### 12.4 Analysis
Let $x^*=\arg\min_xf(x)$.

**Step 1: Jensen.** $f$ convex $\Rightarrow$ $f\big(\frac1T\sum_tx_t\big)\le\frac1T\sum_tf(x_t)$ (the function at an average is at most the average of the function values; "chords lie above a convex function"). So

$$
\mathbb E\big[f(\bar x_T)\big]-f(x^*)\le\frac1T\,\mathbb E\Big[\sum_{t=1}^T\big(f(x_t)-f(x^*)\big)\Big].
$$

**Step 2: convexity (tangent at $x_t$).** $f(x^*)\ge f(x_t)+\langle\nabla f(x_t),x^*-x_t\rangle$, i.e.
$$
f(x_t)-f(x^*)\le\langle\nabla f(x_t),\,x_t-x^*\rangle .
$$

**Step 3: replace the true gradient by the noisy one.** Given the past, $x_t$ is fixed and the fresh sample is independent of the past, so $\mathbb E[\tilde z_t\mid x_t]=\nabla f(x_t)$. Hence
$$
\mathbb E\big[\langle\nabla f(x_t),x_t-x^*\rangle\big]=\mathbb E\big[\langle\tilde z_t,x_t-x^*\rangle\big].
$$
(Rigorous version: conditional expectations / filtrations; the lecture gives the idea.)

**Step 4: regret bound.** Sum over $t$ and apply §12.3:
$$
\mathbb E\big[f(\bar x_T)\big]-f(x^*)\le\frac1T\,\mathbb E\Big[\sum_t\langle\tilde z_t,x_t-x^*\rangle\Big]\le\frac{\sqrt2\,BL\sqrt T}{T}.
$$

$$
\boxed{\ \mathbb E\big[f(\bar x_T)\big]-f(x^*)\le\frac{\sqrt2\,BL}{\sqrt T}=O\Big(\frac1{\sqrt T}\Big)\ }
$$

### 12.5 Takeaways
* **SGD is theoretically sound:** in expectation the averaged iterate converges to the optimum at rate $1/\sqrt T$, even with batch size $1$ out of billions of points.
* Mechanism: **regret $O(\sqrt T)$ $\Rightarrow$ average regret $O(1/\sqrt T)$ $\Rightarrow$ (via Jensen) optimisation error $O(1/\sqrt T)$.** An online-learning tool proving a result for a problem that is not online.
* Constants depend on batch size etc. The result is for the **expected** value: with tiny batches the **variance** of a single run can be large, so larger batches still help in practice.

---

## 13. Where next: Part 2 (partial information)

Part 1 (full-information online learning) ends here. Part 2: **partial information** (e.g. bandits): you only observe the loss of the action you took. Harder, but the ideas developed here (FTRL, regularisers matched to geometry, regret) carry over.

---

## 14. Formula table

| # | Concept | Formula | Notes |
|---|---|---|---|
| 1 | FTRL | $w_t=\arg\min_{w\in S}\big[\sum_{i<t}f_i(w)+R(w)\big]$ | $=$ FTL with $f_0=R$ |
| 2 | Generic bound | $R_T(u)\le R(u)-R(w_1)+\sum_t[f_t(w_t)-f_t(w_{t+1})]$ | $w_1=\arg\min R$, often $R(w_1)=0$ |
| 3 | $L_2$ regulariser | $R(w)=\frac1{2\eta}\lVert w\rVert^2$ | |
| 4 | FTRL-$L_2$, linear loss | $w_t=-\eta\sum_{i<t}z_i$, $\ w_{t+1}=w_t-\eta z_t$ | OGD |
| 5 | Sensitivity term | $f_t(w_t)-f_t(w_{t+1})=\eta\lVert z_t\rVert^2$ | |
| 6 | Regret ($L_2$) | $R_T\le\frac{\lVert u\rVert^2}{2\eta}+\eta\sum\lVert z_t\rVert^2\le\frac{B^2}{2\eta}+\eta L^2T$ | $\lVert z_t\rVert\le L$, $\lVert u\rVert\le B$ |
| 7 | Optimal $\eta$ | $\eta^*=\frac{B}{L\sqrt{2T}}$ | from $-\frac{B^2}{2\eta^2}+L^2T=0$ |
| 8 | Tuned regret | $R_T\le\sqrt2\,BL\sqrt T$ | both terms equal $\frac{BL\sqrt T}{\sqrt2}$ |
| 9 | Hedge via $L_2$-FTRL | $B=1$, $L=\sqrt d$ $\Rightarrow$ $O(\sqrt{Td})$ | vs Hedge $O(\sqrt{T\ln d})$ |
| 10 | Norms on simplex | corners $\lVert e_i\rVert=1$; centre $\lVert\cdot\rVert=1/\sqrt d$ | simplex $\subset$ unit ball |
| 11 | Projected OGD | step, then $w\leftarrow B\,w/\lVert w\rVert$ if $\lVert w\rVert>B$ | regret $O(BL\sqrt T)$ |
| 12 | Entropic regulariser | $R(w)=\frac1\eta\sum_jw_j\ln w_j$ | on $\Delta_d$ |
| 13 | Entropic FTRL solution | $w_{t+1,j}=\frac{e^{-\eta L_{t,j}}}{\sum_ie^{-\eta L_{t,i}}}$ | $L_t=\sum_{i\le t}z_i$ |
| 14 | Hedge recursion | $w_{t+1,j}\propto w_{t,j}e^{-\eta z_{t,j}}$ | FTRL(entropy) $=$ Hedge |
| 15 | Stationarity (KKT) | $L_{t,j}+\frac1\eta(1+\ln w_j)+\lambda-\beta_j=0$ | $\frac{d}{dw}w\ln w=1+\ln w$ |
| 16 | Complementary slackness | $\beta_jw_j=0$ | $w_j>0\Rightarrow\beta_j=0$ |
| 17 | Convexity (subgradient form) | $\forall w\,\exists z\,\forall u:\ f(u)\ge f(w)+\langle u-w,z\rangle$ | $z\in\partial f(w)$; $=\nabla f(w)$ if differentiable |
| 18 | OCO linearisation | $R_T(u)\le\sum_t\langle w_t-u,z_t\rangle$, $z_t\in\partial f_t(w_t)$ | |
| 19 | Online (sub)gradient descent | $w_{t+1}=w_t-\eta z_t$ | regret $\sqrt2BL\sqrt T$ |
| 20 | Jensen | $f\big(\sum\lambda_ix_i\big)\le\sum\lambda_if(x_i)$ | convex $f$, $\lambda\in\Delta$ |
| 21 | SGD step | $x_{t+1}=x_t-\eta\tilde z_t$, $\mathbb E[\tilde z_t\mid x_t]=\nabla f(x_t)$ | output $\bar x_T=\frac1T\sum x_t$ |
| 22 | SGD guarantee | $\mathbb E f(\bar x_T)-f(x^*)\le\frac{\sqrt2BL}{\sqrt T}$ | $O(1/\sqrt T)$ |
| 23 | Rates summary | quadratic/strongly convex $O(\log T)$; linear/convex $O(\sqrt T)$; simplex+entropy $O(\sqrt{T\ln d})$ | |

---

## 15. Quiz preparation tips

### 15.1 Definitions to state precisely
* FTRL; regulariser; "round zero" equivalence to FTL.
* Online gradient descent (and subgradient descent); projected OGD.
* Convex set / convex function (tangent-below definition); subgradient; Lipschitz (bounded gradient).
* Improper learning (learner's class larger than comparator class).
* OCO protocol and regret.
* Stochastic gradient descent and the noisy gradient oracle ($\mathbb E[\tilde z]=\nabla f$).

### 15.2 Derivations likely to be asked
1. **FTRL-$L_2$ closed form** and the OGD recursion (gradient $=0$).
2. **Regret bound** $\frac{B^2}{2\eta}+\eta L^2T$ and $\eta^*$, final $\sqrt2BL\sqrt T$.
3. **Hedge-setting comparison:** compute $L=\sqrt d$, $B=1$; state $\sqrt{Td}$ vs $\sqrt{T\ln d}$.
4. **Entropic FTRL $\Rightarrow$ Hedge** (KKT: stationarity, $\beta_j=0$ on the support, normalise).
5. **OCO linearisation** (tangent inequality, then sum).
6. **SGD guarantee** (Jensen, tangent, unbiasedness, regret).

### 15.3 Common traps
* **$R(w)=\frac1{2\eta}\lVert w\rVert^2$** (not $\frac\eta2$). Larger $\eta$ = *weaker* regularisation = *larger* steps.
* **Learner vs comparator norms:** $\lVert z_t\rVert\le L$ bounds the adversary; $\lVert u\rVert\le B$ bounds only the **comparator**, not necessarily the learner's iterates.
* **$B,L$ for Hedge:** $L=\sqrt d$ because the hypercube's longest vector (all ones) has norm $\sqrt d$; $B=1$ because simplex corners have norm $1$. Do not use $B=\sqrt d$.
* **Entropy sign:** the regulariser is $+\frac1\eta\sum w\ln w$ (**negative** entropy); with the opposite sign the problem is concave.
* **Entropic solution needs the cumulative loss** $L_t$ in the exponent; the recursion multiplies by $e^{-\eta z_t}$ and renormalises.
* **Linearisation uses the gradient at the played point $w_t$**, not at $u$.
* **Direction of the inequality:** convex $\Rightarrow$ tangent below $\Rightarrow$ $f_t(w_t)-f_t(u)\le\langle w_t-u,z_t\rangle$.
* **SGD output is the average**, not the last iterate; the $1/T$ factor in front of the regret turns $\sqrt T$ into $1/\sqrt T$.
* **Convex $\neq$ differentiable:** hinge loss is convex with subgradients at the kink.
* **FTRL on quadratic losses is not needed:** with strongly convex losses FTL already gets $\log T$.
* **$\sqrt T$ for general convex is unimprovable** (linear functions are convex).

### 15.4 Quick numeric practice patterns
* Given $B,L,T$: $\eta^*=\frac{B}{L\sqrt{2T}}$ and regret $\sqrt2BL\sqrt T$. E.g. $B=1,L=1,T=4000$: $\eta^*\approx0.0112$, bound $\approx89.4$.
* Given $d,T$: compare $\sqrt{2dT}$ with $\sqrt{2T\ln d}$ (e.g. $d=1000$, $T=10^4$: $4472$ vs $372$).
* Given cumulative losses $L_t$ and $\eta$: the entropic-FTRL point is $\text{softmax}(-\eta L_t)$.
* SGD error after $T$ steps: $\sqrt2BL/\sqrt T$; to halve it you need $4\times$ more steps.

---

## 16. Self-test questions with answers

**Q1.** Why is FTRL "FTL with a zeroth round"?
**A.** Setting $f_0=R$ and running FTL on $f_0,f_1,\dots$ gives $\arg\min[R+\sum_{i<t}f_i]$.

**Q2.** State the generic FTRL regret bound.
**A.** $R_T(u)\le R(u)-R(w_1)+\sum_t[f_t(w_t)-f_t(w_{t+1})]$.

**Q3.** Compute FTRL with $R(w)=\frac1{2\eta}\lVert w\rVert^2$ on linear losses.
**A.** $w_t=-\eta\sum_{i<t}z_i$, i.e. $w_{t+1}=w_t-\eta z_t$ (online gradient descent).

**Q4.** Derive the regret bound and optimal $\eta$.
**A.** Sensitivity $=\eta\lVert z_t\rVert^2\le\eta L^2$; regret $\le\frac{B^2}{2\eta}+\eta L^2T$; $\eta^*=\frac{B}{L\sqrt{2T}}$; regret $\le\sqrt2BL\sqrt T$.

**Q5.** What is improper learning here?
**A.** The learner's iterates $-\eta\sum z_i$ may leave the $B$-ball while the comparator class is restricted to it.

**Q6.** Why does $L_2$-FTRL do worse than Hedge on the simplex?
**A.** To embed the Hedge game: $L=\sqrt d$, $B=1$, giving $O(\sqrt{Td})$ vs $O(\sqrt{T\ln d})$. The $L_2$ geometry doesn't match probability distributions.

**Q7.** What regulariser fits the simplex, and what does FTRL with it give?
**A.** Negative entropy $\frac1\eta\sum w_j\ln w_j$; the solution is $w_{t+1,j}\propto e^{-\eta L_{t,j}}$, i.e. Hedge.

**Q8.** Sketch the KKT derivation of the entropic solution.
**A.** Lagrangian with $\lambda$ for $\sum w=1$ and $\beta_j\ge0$ for $w_j\ge0$; stationarity $L_{t,j}+\frac1\eta(1+\ln w_j)+\lambda-\beta_j=0$; $w_j>0\Rightarrow\beta_j=0$; so $w_j\propto e^{-\eta L_{t,j}}$; normalise.

**Q9.** Define a convex function without derivatives.
**A.** For every $w$ there is $z$ with $f(u)\ge f(w)+\langle u-w,z\rangle$ for all $u$ (a supporting tangent below the graph).

**Q10.** Reduce OCO to the linear game.
**A.** Let $z_t\in\partial f_t(w_t)$. Convexity gives $f_t(w_t)-f_t(u)\le\langle w_t-u,z_t\rangle$; summing bounds the OCO regret by the linear-game regret with loss vectors $z_t$.

**Q11.** Online gradient descent and its regret?
**A.** $w_{t+1}=w_t-\eta z_t$; regret $\le\sqrt2BL\sqrt T$ with $\eta=\frac{B}{L\sqrt{2T}}$, for convex $L$-Lipschitz losses.

**Q12.** Why can't we beat $\sqrt T$ for general convex adversaries?
**A.** Linear functions are convex, and for linear losses $\Omega(\sqrt T)$ regret is unavoidable. Better rates need curvature (strong convexity $\Rightarrow$ $\log T$).

**Q13.** Prove SGD's $O(1/\sqrt T)$ guarantee in outline.
**A.** Jensen on $\bar x_T$ $\Rightarrow$ average of $f(x_t)-f(x^*)$ $\Rightarrow$ tangent bound by $\langle\nabla f(x_t),x_t-x^*\rangle$ $\Rightarrow$ replace gradient by noisy gradient (unbiased) $\Rightarrow$ OGD regret $\sqrt2BL\sqrt T$ divided by $T$.

**Q14.** Why average the iterates in SGD?
**A.** Jensen is applied to the average; the guarantee is for $f(\bar x_T)$ (the last iterate is not covered by this argument).

**Q15.** What is different between OGD and SGD as algorithms?
**A.** SGD uses a noisy (unbiased) gradient, the same $f$ every round, and outputs an averaged iterate; OGD faces an adaptive sequence of different $f_t$.

---

## 17. Lecture slips / things to double-check

1. **Hedge's rate:** the transcript says "square root of $T$ log $T$" in places. The rate is $O(\sqrt{T\ln d})$ ($d$ = number of experts), as in Weeks 1–4.
2. **"$\sqrt{TdB}$":** the lecture garbles the $L_2$-FTRL rate in Hedge's setting. With $B=1$, $L=\sqrt d$ it is $O(BL\sqrt T)=O(\sqrt{Td})$.
3. **Notation:** the transcript writes the step/regularisation parameter as "ea" ($\eta$). The regulariser $\frac1{2\eta}\lVert w\rVert^2$ is "$\frac1{2\,\text{ea}}$". In the entropic derivation the *normalised* iterate is called $w$ (unlike Hedge's unnormalised weights).
4. **Entropic regulariser sign/wording:** the lecture says "penalise the negative of uncertainty". Precisely $R(w)=+\frac1\eta\sum w\ln w$ is the **negative entropy** (convex).
5. **Losses in $[-1,1]$:** the Hedge comparison uses the hypercube $[-1,1]^d$, whereas earlier Hedge was analysed for $\ell\in[0,B]$. The rate $O(\sqrt{T\ln d})$ is unchanged (only constants and a condition like $\eta\le1$ differ). *Beyond the lecture.*
6. **SGD analysis assumptions not stated aloud:** one needs bounded noisy gradients $\lVert\tilde z_t\rVert\le L$ and $\lVert x^*\rVert\le B$ to invoke $\sqrt2BL\sqrt T$; independence of the fresh sample from the past (filtrations) is only hand-waved. The lecture also writes "$x_{t+1}=x_t-\text{step}\cdot\tilde z$" with the step $\eta$ set as in §5.
7. **Convexity statement at boundaries:** the "$\forall w\,\exists z$" definition is the standard subgradient characterisation for convex functions on convex sets; at boundary points of $S$ a subgradient may fail to exist for some convex functions. Not an issue for $S=\mathbb R^d$ or for the examples in the course. *Beyond the lecture.*
8. **Projected OGD and strongly convex $\log T$ results** are stated without proof; treat them as facts to recall, not to derive.
9. **Transcript noise:** several passages (e.g. the drawing of the tangent-line picture, and the sentence "FTRL with quadratic regularisation... $\sqrt{T d B}$") are partly garbled; the notes give the clean versions.

---

## 18. Runnable verification code

Checks numerically: FTRL-$L_2$ $=$ OGD closed form and the regret bound (random and adversarial sequences, plus the generic $\frac{B^2}{2\eta}+\eta\sum\lVert z_t\rVert^2$ bound), the entropic argmin $=$ softmax (versus a numerical optimiser and random simplex points), the Hedge-recursion identity, the $\sqrt d$ vs $\sqrt{\ln d}$ comparison, and the SGD $O(1/\sqrt T)$ guarantee on a logistic-regression problem. Requires `numpy` and `scipy`. Save as `verify_w6.py` and run `python verify_w6.py`.

```python
"""Runnable sanity checks for SDM Week 6: FTRL, online gradient descent, entropic FTRL = Hedge, OCO linearisation, SGD."""
import numpy as np
from scipy.optimize import minimize
rng = np.random.default_rng(2)

# ---------- 1. FTRL with L2 regulariser on linear losses  ==  online gradient descent ----------
d, T, B, L = 5, 4000, 1.0, 1.0
eta = B / (L * np.sqrt(2 * T))                          # eta* from the lecture
Z = rng.normal(size=(T, d)); Z = L * Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
Z[:, 0] -= 0.1; Z = L * Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), L)   # keep ||z_t|| <= L, add a drift

w = np.zeros(d); S = np.zeros(d); loss = 0.0
for t in range(T):
    w_ftrl = -eta * S                                    # closed form argmin_w <S,w> + ||w||^2/(2 eta)
    assert np.allclose(w, w_ftrl)                        # OGD recursion w_{t+1} = w_t - eta z_t gives the same point
    loss += w @ Z[t]
    w = w - eta * Z[t]; S += Z[t]
u = -B * S / np.linalg.norm(S)                           # best comparator in the ball ||u|| <= B
R = loss - u @ S
print(f"[FTRL-L2 = OGD, random] regret (can be < 0: learner may leave the B-ball) = {R:7.2f} <= sqrt(2) B L sqrt(T) = {np.sqrt(2)*B*L*np.sqrt(T):.2f}")
assert R <= np.sqrt(2) * B * L * np.sqrt(T)

# adversarial sequence: always push against the learner (z_t = L * w_t/||w_t||), so regret is clearly positive
w = np.zeros(d); S = np.zeros(d); loss = 0.0
for t in range(T):
    zt = L * (w / np.linalg.norm(w) if np.linalg.norm(w) > 0 else np.eye(d)[0])
    loss += w @ zt; w = w - eta * zt; S += zt
R = loss - (-B * np.linalg.norm(S))
print(f"[FTRL-L2, adversarial] regret = {R:7.2f} <= sqrt(2) B L sqrt(T) = {np.sqrt(2)*B*L*np.sqrt(T):.2f}")
assert 0 < R <= np.sqrt(2) * B * L * np.sqrt(T)

# generic bound  B^2/(2 eta) + eta * sum ||z_t||^2  for an arbitrary eta
for e in (0.001, 0.01, 0.1):
    w = np.zeros(d); S = np.zeros(d); loss = 0.0
    for t in range(T):
        loss += w @ Z[t]; w = w - e * Z[t]; S += Z[t]
    R = loss - (-B * np.linalg.norm(S))
    bound = B**2 / (2 * e) + e * np.sum(np.linalg.norm(Z, axis=1) ** 2)
    assert R <= bound + 1e-9
print("[FTRL-L2] R_T <= B^2/(2 eta) + eta sum ||z_t||^2 holds for eta in {0.001, 0.01, 0.1}")

# ---------- 2. FTRL with entropic regulariser on the simplex == Hedge ----------
d = 6; eta = 0.3
Lcum = rng.normal(size=d) * 3                            # cumulative loss vector L_t
obj = lambda p: p @ Lcum + (1 / eta) * np.sum(p * np.log(p))
softmax = np.exp(-eta * Lcum) / np.exp(-eta * Lcum).sum()
res = minimize(obj, np.ones(d) / d, method="SLSQP", bounds=[(1e-9, 1)] * d,
               constraints=[{"type": "eq", "fun": lambda p: p.sum() - 1}], options={"ftol": 1e-14, "maxiter": 500})
print("[Entropic FTRL] numeric argmin :", res.x.round(5))
print("[Entropic FTRL] softmax(-eta L):", softmax.round(5))
assert np.allclose(res.x, softmax, atol=1e-4)
assert all(obj(softmax) <= obj(q / q.sum()) + 1e-12 for q in rng.random((2000, d)))   # softmax beats random simplex points

# multiplicative recursion  w_{t+1,j} ∝ w_{t,j} exp(-eta z_{t,j})
z = rng.normal(size=d)
w_next = np.exp(-eta * (Lcum + z)); w_next /= w_next.sum()
w_rec = softmax * np.exp(-eta * z); w_rec /= w_rec.sum()
assert np.allclose(w_next, w_rec)
print("[Entropic FTRL] one-step recursion = Hedge update: OK")

# ---------- 3. L2-FTRL vs Hedge bounds in the Hedge setting (B = 1, L = sqrt(d)) ----------
T = 10_000
print("[bounds] d     L2-FTRL sqrt(2 d T)    Hedge sqrt(2 T ln d)")
for d in (10, 100, 1000, 10_000):
    print(f"         {d:<6d}{np.sqrt(2*d*T):>16.0f}{np.sqrt(2*T*np.log(d)):>20.0f}")

# ---------- 4. SGD: E f(xbar_T) - f* <= sqrt(2) B L / sqrt(T) ----------
n, dim = 500, 5
A = rng.normal(size=(n, dim)); A /= np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1.0)   # ||a_i|| <= 1
yv = np.sign(A @ rng.normal(size=dim) + 0.3 * rng.normal(size=n))                                 # noisy labels
f = lambda x: np.mean(np.log1p(np.exp(-yv * (A @ x))))
grad = lambda x: np.mean((-yv / (1 + np.exp(yv * (A @ x))))[:, None] * A, axis=0)
x = np.zeros(dim)
for _ in range(20000): x -= 0.5 * grad(x)                # near-exact minimiser by full-batch GD
fstar, Bn, Lg = f(x), np.linalg.norm(x), 1.0             # logistic gradient norm <= ||a_i|| <= 1
print(f"[SGD] f* = {fstar:.4f}, B = ||x*|| = {Bn:.2f}, L = {Lg}")
for T in (100, 1000, 10000):
    eta = Bn / (Lg * np.sqrt(2 * T)); gaps = []
    for _ in range(100):
        xt = np.zeros(dim); xs = np.zeros(dim)
        for _t in range(T):
            xs += xt
            i = rng.integers(n)                          # one random sample -> unbiased noisy gradient
            xt = xt - eta * (-yv[i] / (1 + np.exp(yv[i] * (A[i] @ xt)))) * A[i]
        gaps.append(f(xs / T) - fstar)
    print(f"      T={T:<6d} E[f(xbar)]-f* = {np.mean(gaps):.5f} <= sqrt(2) B L/sqrt(T) = {np.sqrt(2)*Bn*Lg/np.sqrt(T):.5f}")
    assert np.mean(gaps) <= np.sqrt(2) * Bn * Lg / np.sqrt(T)
print("All checks passed.")

```

**Expected output**

```text
[FTRL-L2 = OGD, random] regret (can be < 0: learner may leave the B-ball) = -242.51 <= sqrt(2) B L sqrt(T) = 89.44
[FTRL-L2, adversarial] regret =   22.36 <= sqrt(2) B L sqrt(T) = 89.44
[FTRL-L2] R_T <= B^2/(2 eta) + eta sum ||z_t||^2 holds for eta in {0.001, 0.01, 0.1}
[Entropic FTRL] numeric argmin : [0.17227 0.13939 0.06523 0.20624 0.1136  0.30327]
[Entropic FTRL] softmax(-eta L): [0.17227 0.13939 0.06523 0.20624 0.1136  0.30327]
[Entropic FTRL] one-step recursion = Hedge update: OK
[bounds] d     L2-FTRL sqrt(2 d T)    Hedge sqrt(2 T ln d)
         10                 447                 215
         100               1414                 303
         1000              4472                 372
         10000            14142                 429
[SGD] f* = 0.3726, B = ||x*|| = 5.73, L = 1.0
      T=100    E[f(xbar)]-f* = 0.11035 <= sqrt(2) B L/sqrt(T) = 0.81000
      T=1000   E[f(xbar)]-f* = 0.03023 <= sqrt(2) B L/sqrt(T) = 0.25615
      T=10000  E[f(xbar)]-f* = 0.00460 <= sqrt(2) B L/sqrt(T) = 0.08100
All checks passed.
```

---
*End of Week 6 notes (end of Part 1: full-information online learning).*
