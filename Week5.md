# Sequential Decision Making — Week 5 Notes
**General Online Learning, Follow-the-Leader (FTL), Be-the-Leader (BTL), Quadratic vs Linear Adversaries**

> Continuation of the Weeks 1–4 notes (experts, regret, Hedge, combinatorial Hedge).
> Built from lecture transcript L24–29. Math is GitHub-flavoured LaTeX (`$...$`, `$$...$$`).
> Notation: $S$ = action set, $w_t\in S$ = learner's play, $f_t:S\to\mathbb R$ = loss function revealed in round $t$, $\lVert\cdot\rVert$ = Euclidean norm.

---

## Table of Contents
1. [One-page summary](#1-one-page-summary)
2. [Clarification from last lecture: Sinkhorn is a KL projection](#2-clarification-from-last-lecture-sinkhorn-is-a-kl-projection)
3. [The general online learning protocol](#3-the-general-online-learning-protocol)
4. [Hedge and online regression as special cases](#4-hedge-and-online-regression-as-special-cases)
5. [Regret in the general setting](#5-regret-in-the-general-setting)
6. [Follow the Leader (FTL) and Be the Leader (BTL)](#6-follow-the-leader-ftl-and-be-the-leader-btl)
7. [Key claim: BTL beats every fixed point (proof by induction)](#7-key-claim-btl-beats-every-fixed-point)
8. [Reduction: regret of FTL $\le$ FTL–BTL gap](#8-reduction-regret-of-ftl--ftlbtl-gap)
9. [Quadratic adversary: FTL gets $O(\log T)$ regret](#9-quadratic-adversary-ftl-gets-olog-t-regret)
10. [Linear adversary: FTL fails (linear regret)](#10-linear-adversary-ftl-fails-linear-regret)
11. [Why the two cases differ: stability of the leader](#11-why-the-two-cases-differ-stability-of-the-leader)
12. [Where next: regularization and FTRL](#12-where-next-regularization-and-ftrl)
13. [Formula table](#13-formula-table)
14. [Quiz preparation tips](#14-quiz-preparation-tips)
15. [Self-test questions with answers](#15-self-test-questions-with-answers)
16. [Lecture slips / things to double-check](#16-lecture-slips--things-to-double-check)
17. [Runnable verification code](#17-runnable-verification-code)

---

## 1. One-page summary

| Idea | Statement |
|---|---|
| **General protocol** | Learner plays $w_t\in S$; adversary reveals a **function** $f_t:S\to\mathbb R$; learner suffers $f_t(w_t)$. Full information (the whole function is shown). |
| **Hedge is a special case** | $S=\Delta_d$, $f_t(p)=\langle p,\ell_t\rangle$ (linear). |
| **Online ML is a special case** | $S=\mathbb R^d$, e.g. $f_t(w)=(w^\top x_t-y_t)^2$ (online regression); swap the loss for logistic, hinge, exponential, etc. |
| **Regret** | $R_T=\sum_t f_t(w_t)-\min_{u\in S}\sum_t f_t(u)$ |
| **FTL** | $w_t=\arg\min_{w\in S}\sum_{i=1}^{t-1}f_i(w)$ — "do the best on what you have seen" |
| **BTL** (hypothetical, cannot be run) | $w_t^{\text{BTL}}=\arg\min_{w\in S}\sum_{i=1}^{t}f_i(w)$ — peeks at $f_t$ |
| **BTL lemma** | $\sum_t f_t(w_{t+1}^{\text{FTL}})\le\sum_t f_t(u)\ \ \forall u\in S$ (BTL beats every fixed point) |
| **Reduction** | $R_T(\text{FTL})\le\sum_t\big[f_t(w_t^{\text{FTL}})-f_t(w_{t+1}^{\text{FTL}})\big]$ = "how sensitive is FTL to one extra function?" |
| **Quadratic adversary** $f_t(w)=\tfrac12\lVert w-z_t\rVert^2$, $\lVert z_t\rVert\le L$ | FTL plays the **running average** of the $z$'s $\Rightarrow$ gap per round $\le 4L^2/t$ $\Rightarrow$ $R_T\le 4L^2(1+\ln T)=\mathbf{O(\log T)}$ |
| **Linear adversary** $f_t(w)=z_tw$ on $[-1,1]$ | FTL jumps between the endpoints $\pm1$; gap is $2$ per round $\Rightarrow$ $R_T=\Omega(T)$: **FTL is useless** |
| **Reason** | Averages are *stable* (one new point moves the average by $O(1/t)$); minimisers of linear functions sit at corners and *flip* |
| **Fix (next lecture)** | **Regularise**: Follow the **Regularised** Leader (FTRL) |

**Five ideas to remember**
1. The adversary now reveals an entire **function**, which covers Hedge and any online ML problem.
2. **Compare to BTL, not to the best fixed point:** BTL is *better* than every fixed $u$, so bounding FTL vs BTL also bounds FTL vs $u$.
3. Regret of FTL is controlled by its **sensitivity** to one extra function.
4. **Quadratic loss $\Rightarrow$ log regret** (strong curvature makes the adversary weak). **Linear loss $\Rightarrow$ FTL fails** (Hedge/$\sqrt T$ needed).
5. Instability is the disease; **regularisation** is the cure.

---

## 2. Clarification from last lecture: Sinkhorn is a KL projection

Last week it was said that Sinkhorn balancing solves
$$\min_{P\in\text{DSP}}\lVert W-P\rVert_F^2 .$$
**Correction from the professor:** this is *not* the objective Sinkhorn minimises. The closeness measure is a **relative-entropy (KL)** based distance, not the Frobenius/Euclidean norm. The idea is unchanged: Sinkhorn finds a point of the polytope that is "closest" to $W$, only under a different notion of closeness. (The same holds for normalising a weight vector onto the simplex in Hedge.)

> Already folded into §14.7 and §18 of the Weeks 1–4 notes.

---

## 3. The general online learning protocol

For $t=1,\dots,T$:
1. Learner **predicts** (plays) $w_t\in S$.
2. Adversary reveals a **function** $f_t:S\to\mathbb R$.
3. Learner **suffers** $f_t(w_t)$.

* $S$ is the action space (also called the *model space*; "predict" foreshadows ML where $w_t$ is a model).
* **Full information:** $f_t$ is shown on *all* of $S$, so you learn what *every* alternative $w\in S$ would have cost in round $t$.
* So far (Hedge): $S=\Delta_d$ and $f_t$ linear. Now: **general $S$, general $f_t$**.
* The adversary may pick a different *form* each round (quadratic, then logistic, then cubic, ...). In ML we usually restrict to **one functional form with changing parameters**.

---

## 4. Hedge and online regression as special cases

### 4.1 Hedge
$w_t=p_t\in S=\Delta_d$. The loss vector $\ell_t$ defines the function
$$f_t(p)=p^\top\ell_t=\langle p,\ell_t\rangle\qquad\forall p\in\Delta_d .$$
So $\ell_t$ tells you the loss of **every** $p$, not only of the $p_t$ you played. This is exactly what "full information" means.

### 4.2 Online regression
Standard online protocol: receive $x_t\in\mathbb R^d$, predict $\hat y_t=w_t^\top x_t$, receive label $y_t\in\mathbb R$, suffer $(w_t^\top x_t-y_t)^2$.

**Matching it to the general protocol.** Take $S=\mathbb R^d$. Equivalent views:
* The learner announces $w_t$; then nature chooses $(x_t,y_t)$ (the "data point arrives after the model is fixed"), which defines
$$f_t(w)=(w^\top x_t-y_t)^2 .$$
* The loss suffered is $f_t(w_t)=(w_t^\top x_t-y_t)^2$.

$f_t$ is a **quadratic** function of $w$; only its parameters $(x_t,y_t)$ change from round to round.

**Other ML problems = other loss shapes:** logistic loss $\to$ online logistic regression; hinge loss $\to$ online SVM-type problem; exponential loss $\to$ online boosting.

### 4.3 Why the general protocol is worth studying
Any algorithm with a regret bound for **general** $f_t$ is a master algorithm for all of the above. But a fully general adversary seems too powerful, so the plan is:
1. See how far we get with **no** assumption on $f_t$.
2. Find where it breaks.
3. **Restrict the adversary** (e.g. quadratic/convex/linear functions) and see which regret bounds become possible.

---

## 5. Regret in the general setting

$$
\boxed{\ R_T=\sum_{t=1}^{T}f_t(w_t)-\min_{u\in S}\sum_{t=1}^{T}f_t(u)\ }
$$

* Same idea as before: compare with the **best fixed point in hindsight**.
* **Online regression:**
$$R_T=\sum_{t=1}^T(w_t^\top x_t-y_t)^2-\min_{u\in\mathbb R^d}\sum_{t=1}^T(u^\top x_t-y_t)^2 .$$
The second term is exactly **offline linear regression** on the data $(x_1,y_1),\dots,(x_T,y_T)$.
* **Meaning of sublinear regret here:** an algorithm that never sees the data in advance still matches the best offline least-squares model on average, $R_T/T\to0$. That is the "good news" of online learning.

---

## 6. Follow the Leader (FTL) and Be the Leader (BTL)

### 6.1 FTL: the simplest algorithm
In round $t$ you know $f_1,\dots,f_{t-1}$. The most natural move: pick the point that does best on everything seen so far.

$$
\boxed{\ w_t^{\text{FTL}}=\arg\min_{w\in S}\sum_{i=1}^{t-1}f_i(w)\ }
$$

(For $t=1$ the sum is empty, so $w_1$ is any point of $S$.)

### 6.2 BTL: the hypothetical algorithm
If the adversary revealed $f_t$ **before** you play, you could use the current round too:

$$
w_t^{\text{BTL}}=\arg\min_{w\in S}\sum_{i=1}^{t}f_i(w).
$$

BTL **cannot be run** (it needs $f_t$ before choosing). It is only an analysis tool.

### 6.3 Why "follow"
$$w_{t+1}^{\text{FTL}}=\arg\min_{w}\sum_{i=1}^{t}f_i(w)=w_t^{\text{BTL}} .$$
FTL's choice in round $t+1$ is BTL's choice in round $t$: FTL always copies BTL **one round late**. So

$$
\text{loss of BTL}=\sum_{t=1}^Tf_t\big(w_t^{\text{BTL}}\big)=\sum_{t=1}^Tf_t\big(w_{t+1}^{\text{FTL}}\big).
$$

> **Notation trap.** In the lecture, "$w_{t+1}$" in $f_t(w_{t+1})$ always means **FTL's** next iterate $=$ BTL's current iterate.

---

## 7. Key claim: BTL beats every fixed point

> **Claim (BTL lemma).** For every $u\in S$ and every $T$:
> $$\sum_{t=1}^{T}f_t\big(w_{t+1}^{\text{FTL}}\big)\ \le\ \sum_{t=1}^{T}f_t(u).$$

Even though BTL is a *moving* sequence of points, its total loss is no larger than that of any *single* point. (Reason for intuition: BTL gets to peek at $f_t$.)

### Proof by induction on $T$

**Base case $T=1$.** $w_2^{\text{FTL}}=\arg\min_wf_1(w)$, so $f_1(w_2)\le f_1(u)$ for every $u\in S$. ✔

**Induction step.** Assume the claim for $T-1$:
$$\sum_{t=1}^{T-1}f_t(w_{t+1})\le\sum_{t=1}^{T-1}f_t(u)\qquad\forall u\in S.\tag{A}$$
Add $f_T(w_{T+1})$ to both sides:
$$\sum_{t=1}^{T}f_t(w_{t+1})\le f_T(w_{T+1})+\sum_{t=1}^{T-1}f_t(u)\qquad\forall u .$$
(A) holds for *every* $u$, in particular for $u=w_{T+1}\in S$. Substituting it:
$$\sum_{t=1}^{T}f_t(w_{t+1})\le f_T(w_{T+1})+\sum_{t=1}^{T-1}f_t(w_{T+1})=\sum_{t=1}^{T}f_t(w_{T+1}).\tag{B}$$
But $w_{T+1}=\arg\min_w\sum_{t=1}^Tf_t(w)$ by definition, so
$$\sum_{t=1}^{T}f_t(w_{T+1})\le\sum_{t=1}^{T}f_t(u)\qquad\forall u\in S.\tag{C}$$
Chain (B) and (C): $\sum_{t\le T}f_t(w_{t+1})\le\sum_{t\le T}f_t(u)$. $\blacksquare$

**Proof pattern to memorise:** induct; add the new term to both sides; choose the special $u=w_{T+1}$; use the argmin property of $w_{T+1}$.

---

## 8. Reduction: regret of FTL $\le$ FTL–BTL gap

Add and subtract BTL's loss. Let $u^*$ be the best fixed point.

$$
R_T(\text{FTL})=\underbrace{\sum_tf_t(w_t)-\sum_tf_t(w_{t+1})}_{\text{FTL loss}-\text{BTL loss}}+\underbrace{\sum_tf_t(w_{t+1})-\sum_tf_t(u^*)}_{\le\,0\ \text{by the BTL lemma}} .
$$

Therefore

$$
\boxed{\ R_T(\text{FTL})\ \le\ \sum_{t=1}^{T}\Big[f_t\big(w_t^{\text{FTL}}\big)-f_t\big(w_{t+1}^{\text{FTL}}\big)\Big]\ }
$$

**Interpretation: sensitivity.** FTL uses $f_1..f_{t-1}$; BTL uses $f_1..f_t$ (**one extra function**). The bracket measures how much *one extra function* changes the algorithm's choice (and loss).
* If FTL is **very sensitive** (a single function can swing $w$ a lot), the adversary can make the gap large each round $\Rightarrow$ bad regret.
* If FTL is **stable**, the gap is small $\Rightarrow$ good regret.

The next two sections test this on a quadratic and a linear adversary.

---

## 9. Quadratic adversary: FTL gets $O(\log T)$ regret

### 9.1 Setting
* $S=\mathbb R^d$ (any vector allowed for the learner).
* Adversary picks $z_t$ with $\lVert z_t\rVert\le L$ (a ball of radius $L$) and the loss is
$$f_t(w)=\tfrac12\lVert w-z_t\rVert^2 .$$
(The $\tfrac12$ is only a normalising constant.)

### 9.2 What FTL plays: the running average
$$
w_t^{\text{FTL}}=\arg\min_w\sum_{i=1}^{t-1}\tfrac12\lVert w-z_i\rVert^2 .
$$
Set the gradient to zero: $\sum_{i<t}(w-z_i)=0$, so

$$
\boxed{\ w_t^{\text{FTL}}=\frac1{t-1}\sum_{i=1}^{t-1}z_i\ }\qquad
w_{t+1}^{\text{FTL}}=\frac1t\sum_{i=1}^{t}z_i .
$$

Geometrically, it is the **centroid** of the points seen so far (the point minimising the total squared distance).

Both $w_t^{\text{FTL}}$ and $w_{t+1}^{\text{FTL}}$ are **averages = convex combinations** of points in the ball of radius $L$, so they also lie in that ball: $\lVert w_t^{\text{FTL}}\rVert\le L$.

### 9.3 BTL as a moving average of FTL
$$
w_{t+1}=\frac1t\Big(\sum_{i=1}^{t-1}z_i+z_t\Big)=\frac{t-1}{t}\,w_t+\frac1t\,z_t .
$$
Only a $1/t$ weight goes to the new point: the extra function can pull the iterate **only a little, and less and less as $t$ grows**.

### 9.4 Per-round gap
From $w_{t+1}=\frac{t-1}{t}w_t+\frac1tz_t$:
$$
w_{t+1}-z_t=\frac{t-1}{t}\,(w_t-z_t).
$$
Hence
$$
f_t(w_t)-f_t(w_{t+1})=\tfrac12\lVert w_t-z_t\rVert^2\Big(1-\frac{(t-1)^2}{t^2}\Big)=\tfrac12\lVert w_t-z_t\rVert^2\cdot\frac{2t-1}{t^2}.
$$
Using $2t-1\le2t$:
$$
f_t(w_t)-f_t(w_{t+1})\ \le\ \frac1t\,\lVert w_t-z_t\rVert^2 .
$$
Since $\lVert w_t\rVert\le L$ and $\lVert z_t\rVert\le L$, the triangle inequality gives $\lVert w_t-z_t\rVert\le2L$ (two points of the ball are at most a diameter apart), so $\lVert w_t-z_t\rVert^2\le4L^2$:

$$
\boxed{\ f_t(w_t)-f_t(w_{t+1})\le\frac{4L^2}{t}\ }
$$

### 9.5 Summing over rounds: harmonic series
$$
R_T(\text{FTL})\le\sum_{t=1}^T\frac{4L^2}{t}=4L^2H_T\le4L^2\,(1+\ln T),
$$
where $H_T=\sum_{t=1}^T\frac1t$ is the harmonic number.

$$
\boxed{\ R_T(\text{FTL, quadratic adversary})=O\big(L^2\log T\big)\ }
$$

### 9.6 Interpretation
* **Logarithmic** regret, far better than Hedge's $\sqrt T$ for linear losses. Average regret $R_T/T=O(\log T/T)\to0$ very fast, so FTL is a **no-regret** algorithm here.
* The adversary is **weak**: to hurt you it must make *your* loss large while keeping *some fixed point's* loss small. With quadratic losses it cannot do both at once. (Intuition: "an average is a robust quantity; adding one point can't move it much.")
* **Caution:** the comparison is of FTL against BTL, and BTL is even better than the best fixed point, so the true regret is at most this bound (it may be smaller).
* The setting differs from Hedge's (unconstrained $S=\mathbb R^d$ with curved loss vs. simplex with linear loss), so the better rate does **not** make Hedge obsolete (see §10–11).

---

## 10. Linear adversary: FTL fails (linear regret)

### 10.1 Setting (1-D)
* $S=[-1,1]$, adversary picks $z_t\in[-1,1]$, loss $f_t(w)=z_t\,w$ (linear in $w$).
* FTL: $w_t=\arg\min_{w\in[-1,1]}\ w\sum_{i<t}z_i$. A linear function over an interval is minimised at an **endpoint**: $w_t=-\mathrm{sign}\big(\sum_{i<t}z_i\big)$ (and $0$, say, if the sum is $0$, e.g. $t=1$).
* BTL: same, but with $\sum_{i\le t}z_i$.

### 10.2 The adversary's strategy (adversary hat)
Whenever FTL plays $w_t$, set $z_t=w_t$ (same sign, magnitude 1) so FTL suffers loss $+1$. Start with $z_1=0.5$ (FTL plays $0$ in round 1).

| Round $t$ | $w_t^{\text{FTL}}$ | $z_t$ | $w_t^{\text{BTL}}$ | FTL loss $z_tw^{\text{FTL}}_t$ | BTL loss $z_tw^{\text{BTL}}_t$ |
|---|---|---|---|---|---|
| 1 | $0$ | $0.5$ | $-1$ | $0$ | $-0.5$ |
| 2 | $-1$ | $-1$ | $+1$ | $+1$ | $-1$ |
| 3 | $+1$ | $+1$ | $-1$ | $+1$ | $-1$ |
| 4 | $-1$ | $-1$ | $+1$ | $+1$ | $-1$ |
| $\vdots$ | alternates | alternates | alternates | $+1$ | $-1$ |

Why BTL flips: cumulative sums of $z$ go $0.5,\,-0.5,\,0.5,\,-0.5,\dots$, so the minimiser of $w\cdot(\text{sum})$ alternates between $-1$ and $+1$; FTL copies BTL one round late, which is exactly the **wrong** side each time because the adversary reacts.

### 10.3 Consequence
* After $T$ rounds: FTL's cumulative loss $=T-1$; BTL's cumulative loss $=-T+0.5$.
* Gap $=(T-1)-(-T+0.5)=2T-1.5=\Theta(T)$.

$$
\text{upper bound from FTL vs BTL}=\Omega(T)\ \Longrightarrow\ \text{this proof technique gives nothing.}
$$

**It is not just a weak proof.** The lecture states (without proof) that FTL really does have linear regret on linear losses. In this example: the best fixed $u$ has loss $\approx-|\sum z_t|=-0.5$, so $R_T\approx T-0.5$ (checked in the code).

### 10.4 Conclusion
$$
\text{FTL}:\ \ \text{quadratic adversary}\Rightarrow O(\log T)\qquad\text{linear adversary}\Rightarrow\Theta(T)\ (\text{useless}).
$$

---

## 11. Why the two cases differ: stability of the leader

| | Quadratic losses | Linear losses |
|---|---|---|
| What FTL plays | the **average** of $z_1..z_{t-1}$ (interior point) | an **extreme point** (endpoint of the interval / corner of the polytope) |
| Effect of one extra function | moves $w$ by at most $\frac1t\lVert z_t-w_t\rVert$ | can flip $w$ from one extreme to the other |
| FTL–BTL gap per round | $\le4L^2/t\to0$ | $=2$, **constant** |
| Total | $O(\log T)$ | $\Theta(T)$ |
| Verdict | **stable** | **unstable / very sensitive** |

* After $100$ rounds the FTL point is the average of $99$ points and BTL's is the average of $100$: the new point has weight $\frac1{100}$. After $1000$ rounds, weight $\frac1{1000}$. The average only gets harder to move.
* Picture (1-D): quadratics $\frac12(w-z_i)^2$ are parabolas whose sum has a unique, smoothly-moving minimiser (the mean); linear functions $z_iw$ are lines whose sum is again a line, minimised at an endpoint that jumps when the slope changes sign.

**Moral.** FTL's weakness is **instability**. For quadratic losses stability comes for free (curvature). For linear losses it must be *forced* by the algorithm.

---

## 12. Where next: regularization and FTRL

* Idea from ML: when you want a model to have a property (here: *not move much*), **add a regulariser** to the objective.
* **Follow the Regularised Leader (FTRL):**
$$
w_t=\arg\min_{w\in S}\Big[\sum_{i=1}^{t-1}f_i(w)+\text{regulariser}(w)\Big].
$$
(Only previewed in the lecture; its regret analysis is for the next lecture.)
* Teaser questions: which regulariser? Does FTRL get good regret for **linear** losses? Can it reach $\log T$ like FTL on quadratics (which would make Hedge unnecessary)? **Do not assume the answers**; they were left open.
* Background reminder: for linear losses on the simplex a $\Omega(\sqrt T)$ lower bound already exists (Weeks 1–4), so $\log T$ for linear losses on the simplex is not possible; the $\log T$ result needs curvature.

---

## 13. Formula table

| # | Concept | Formula | Notes |
|---|---|---|---|
| 1 | General protocol | play $w_t\in S$; see $f_t:S\to\mathbb R$; suffer $f_t(w_t)$ | full information |
| 2 | Hedge as special case | $S=\Delta_d$, $f_t(p)=\langle p,\ell_t\rangle$ | linear |
| 3 | Online regression loss | $f_t(w)=(w^\top x_t-y_t)^2$ | quadratic in $w$; params $(x_t,y_t)$ |
| 4 | Regret | $R_T=\sum_tf_t(w_t)-\min_{u\in S}\sum_tf_t(u)$ | vs best fixed point |
| 5 | FTL | $w_t=\arg\min_{w\in S}\sum_{i<t}f_i(w)$ | $w_1$ arbitrary |
| 6 | BTL | $w_t^{\text{BTL}}=\arg\min_{w\in S}\sum_{i\le t}f_i(w)=w_{t+1}^{\text{FTL}}$ | not runnable |
| 7 | BTL lemma | $\sum_tf_t(w_{t+1})\le\sum_tf_t(u)\ \forall u$ | induction |
| 8 | Reduction | $R_T(\text{FTL})\le\sum_t[f_t(w_t)-f_t(w_{t+1})]$ | sensitivity to one extra function |
| 9 | Quadratic adversary | $f_t(w)=\tfrac12\lVert w-z_t\rVert^2$, $\lVert z_t\rVert\le L$ | |
| 10 | FTL (quadratic) | $w_t=\frac1{t-1}\sum_{i<t}z_i$ | running mean |
| 11 | BTL via FTL | $w_{t+1}=\frac{t-1}{t}w_t+\frac1tz_t$ | moving average |
| 12 | Exact gap | $f_t(w_t)-f_t(w_{t+1})=\tfrac12\lVert w_t-z_t\rVert^2\,\frac{2t-1}{t^2}$ | uses $w_{t+1}-z_t=\frac{t-1}t(w_t-z_t)$ |
| 13 | Gap bound | $\le\frac1t\lVert w_t-z_t\rVert^2\le\frac{4L^2}t$ | $\lVert w_t-z_t\rVert\le2L$ |
| 14 | Regret (quadratic) | $R_T\le4L^2H_T\le4L^2(1+\ln T)=O(\log T)$ | $H_T=\sum_{t\le T}1/t$ |
| 15 | Linear 1-D game | $S=[-1,1]$, $f_t(w)=z_tw$ | FTL plays endpoints |
| 16 | FTL–BTL gap (linear) | $=2T-1.5=\Theta(T)$ | per-round gap 2 |
| 17 | Regret (linear, FTL) | $\Theta(T)$ | no-regret fails |
| 18 | Harmonic bounds | $\ln(T+1)\le H_T\le1+\ln T$ | |
| 19 | Regret rates seen so far | experts/Hedge (linear): $\Theta(\sqrt{T\ln d})$; FTL + quadratic: $O(\log T)$; FTL + linear: $\Theta(T)$ | |

---

## 14. Quiz preparation tips

### 14.1 Definitions to state precisely
* General online learning protocol (play $w_t$, receive function $f_t$, suffer $f_t(w_t)$).
* Regret for general $f_t$; **offline optimum = best fixed $u\in S$ in hindsight**.
* **FTL**, **BTL** (and that BTL is not implementable), and the relation $w_t^{\text{BTL}}=w_{t+1}^{\text{FTL}}$.
* Quadratic adversary (ball of radius $L$) and linear adversary.

### 14.2 Proofs likely to be asked
1. **BTL lemma by induction** (base case, add $f_T(w_{T+1})$, plug $u=w_{T+1}$, use argmin).
2. **Regret reduction** (add and subtract BTL's loss; drop the non-positive term).
3. **Quadratic case:** FTL $=$ mean $\Rightarrow$ BTL $=\frac{t-1}tw_t+\frac1tz_t\Rightarrow$ gap $\le4L^2/t\Rightarrow$ harmonic sum.
4. **Linear 1-D counter-example:** fill in the round-by-round table (FTL, $z_t$, BTL, losses) and compute the cumulative gap.

### 14.3 Common traps
* **FTL's argmin runs over $i\le t-1$**, BTL's over $i\le t$. Mixing up the indices is the most common error.
* **FTL regret bound is against BTL**, which is *at least as good as* the best fixed point. So the bound is valid but the argument goes through an inequality, not an equality.
* **Gap algebra:** $1-\big(\frac{t-1}{t}\big)^2=\frac{2t-1}{t^2}$ (not $\frac1{t}$); the bound $\frac{2t-1}{2t^2}\le\frac1t$ discards the $-1$.
* **Diameter step:** $\lVert w_t-z_t\rVert\le2L$ needs *both* points to lie in the $L$-ball (FTL's point does, since it is an average).
* **Harmonic sum:** $\sum_{t\le T}\frac1t\le1+\ln T$ (the lecture writes $\ln T$ or $\ln(T+1)$ loosely). Either way it is $\Theta(\log T)$.
* **Linear ≠ quadratic:** do not claim FTL is good in general. It is good for quadratic (strongly curved) losses, bad for linear.
* **Why not drop Hedge?** The $\log T$ result is for a different setting (quadratic losses). For linear losses on the simplex, $\Omega(\sqrt T)$ is unavoidable.
* **FTL at $t=1$:** the argmin over an empty sum is arbitrary (lecture picks $0$).

### 14.4 Quick numeric practice patterns
* Given $L$ and $T$: bound $=4L^2(1+\ln T)$. E.g. $L=2,\,T=5000$: $\approx152$.
* Given $z_1,\dots,z_{t-1}$: FTL's play is their mean; BTL's is the mean including $z_t$.
* Given a linear sequence on $[-1,1]$: FTL plays $-\mathrm{sign}(\text{running sum})$; BTL plays $-\mathrm{sign}(\text{running sum incl. current})$.
* Verify: FTL loss $=T-1$, BTL loss $=-T+0.5$ for the lecture's sequence.

---

## 15. Self-test questions with answers

**Q1.** Write Hedge in the general protocol.
**A.** $S=\Delta_d$, $w_t=p_t$, $f_t(p)=\langle p,\ell_t\rangle$.

**Q2.** Write online linear regression in the general protocol.
**A.** $S=\mathbb R^d$, $f_t(w)=(w^\top x_t-y_t)^2$, regret vs. the offline least-squares solution.

**Q3.** Define FTL and BTL. How are they related?
**A.** FTL minimises $\sum_{i<t}f_i$; BTL minimises $\sum_{i\le t}f_i$. $w^{\text{BTL}}_t=w^{\text{FTL}}_{t+1}$.

**Q4.** Why is BTL not a real algorithm?
**A.** It needs $f_t$ before choosing $w_t$, but the protocol reveals $f_t$ afterwards.

**Q5.** State and prove why BTL's cumulative loss is at most that of any fixed $u$.
**A.** See §7 (induction; use $u=w_{T+1}$).

**Q6.** Why is $R_T(\text{FTL})\le\sum_t[f_t(w_t)-f_t(w_{t+1})]$?
**A.** Add/subtract $\sum_tf_t(w_{t+1})$; the remainder $\sum f_t(w_{t+1})-\min_u\sum f_t(u)\le0$ by the BTL lemma.

**Q7.** What does FTL play for $f_t(w)=\tfrac12\lVert w-z_t\rVert^2$?
**A.** The mean of $z_1,\dots,z_{t-1}$.

**Q8.** Derive the per-round gap in the quadratic case.
**A.** $w_{t+1}-z_t=\frac{t-1}t(w_t-z_t)\Rightarrow$ gap $=\tfrac12\lVert w_t-z_t\rVert^2\frac{2t-1}{t^2}\le\frac1t\lVert w_t-z_t\rVert^2\le\frac{4L^2}t$.

**Q9.** What is FTL's regret bound for the quadratic adversary?
**A.** $4L^2H_T\le4L^2(1+\ln T)=O(\log T)$.

**Q10.** In the linear 1-D example, what is FTL's cumulative loss after $T$ rounds, and why is the FTL–BTL gap linear?
**A.** FTL: $T-1$; BTL: $-T+0.5$; gap $2T-1.5$. FTL and BTL swap endpoints every round, so the per-round gap is a constant $2$.

**Q11.** Why can't the adversary do the same to FTL with quadratic losses?
**A.** FTL plays an average, and one new point has weight only $1/t$; the average stays within the $L$-ball and moves by $O(1/t)$.

**Q12.** If FTL gets $O(\log T)$ on quadratics, why do we still need Hedge?
**A.** Different setting. Linear losses on the simplex need $\Omega(\sqrt T)$ regret regardless, and FTL has linear regret there. Hedge achieves $O(\sqrt{T\ln d})$.

**Q13.** What is the remedy for FTL's instability, and what is its name?
**A.** Regularise the objective: Follow the Regularised Leader (FTRL).

---

## 16. Lecture slips / things to double-check

1. **Sinkhorn projection** (see §2): KL/relative-entropy, not Frobenius.
2. **Harmonic bound:** the lecture says $\sum_{t\le T}\frac1t\le\ln(T+1)$ or $\ln T$ "with constants". The correct statement is $\ln(T+1)\le H_T\le1+\ln T$, so the safe upper bound is $4L^2(1+\ln T)$ (I used this in the notes and the code).
3. **Notation drift:** the transcript writes "$w_{t+1}$" for both FTL's next iterate and BTL's current one, and occasionally mixes "$w_t$ BTL" with "$w_{t+1}$ FTL". They are the same object (§6.3).
4. **Exact gap vs bound:** the lecture throws away a $-\frac1{t^2}$ term ("removing it") to get $\frac1t$; this is a valid upper bound, not an equality.
5. **"Zero-regret" wording:** the lecture calls FTL a "zero-regret algorithm" for quadratic losses; precisely, it has **no-regret / sublinear** regret ($R_T/T\to0$), not $R_T=0$.
6. **Linear-loss claim:** that FTL has *true* linear regret (not merely a loose bound) is stated without proof in the lecture. The 1-D example computed here does show $R_T\approx T-0.5$ (see code), but a general proof was not given.
7. **Garbled transcript passages:** the stretch describing the picture of quadratics vs linear functions is partly unintelligible; §11 gives the clean version.

---

## 17. Runnable verification code

Checks numerically: the FTL regret bound for the quadratic adversary (random and adversarial inputs), the exact per-round gap bound, the BTL lemma, the lecture's linear 1-D counter-example (gap $2T-1.5$ and linear regret), and Hedge as a special case. Requires `numpy`. Save as `verify_w5.py` and run `python verify_w5.py`.

```python
"""Runnable sanity checks for SDM Week 5: general online learning, FTL, BTL, quadratic vs linear adversary."""
import numpy as np
rng = np.random.default_rng(1)

# ---------- 1. FTL on quadratic losses f_t(w) = 1/2 ||w - z_t||^2, ||z_t|| <= L ----------
def ftl_quadratic(Z):
    """Returns (regret, per-round gaps f_t(w_t^FTL) - f_t(w_t^BTL)). w_1 = 0 (mean of empty set)."""
    T, d = Z.shape
    S = np.zeros(d); regret_alg = 0.0; gaps = []
    for t in range(1, T + 1):
        w_ftl = S / (t - 1) if t > 1 else np.zeros(d)      # average of z_1..z_{t-1}
        S = S + Z[t - 1]
        w_btl = S / t                                        # average of z_1..z_t
        f = lambda w: 0.5 * np.sum((w - Z[t - 1]) ** 2)
        regret_alg += f(w_ftl); gaps.append(f(w_ftl) - f(w_btl))
    u = Z.mean(0)                                            # best fixed point in hindsight
    best = 0.5 * np.sum((Z - u) ** 2)
    return regret_alg - best, np.array(gaps)

L, d, T = 2.0, 3, 5000
Z = rng.normal(size=(T, d)); Z = L * Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9) * rng.random((T, 1))
R, gaps = ftl_quadratic(Z)
print(f"[FTL quad, random]      regret = {R:8.2f} <= 4L^2(1+ln T) = {4*L**2*(1+np.log(T)):.2f}")
assert R <= 4 * L**2 * (1 + np.log(T))

# near-worst case in 1-D: adversary always places z_t at +-L, opposite side of FTL's current point
z, S, T2 = [], 0.0, 5000
for t in range(1, T2 + 1):
    w = S / (t - 1) if t > 1 else 0.0
    zt = -L if w >= 0 else L
    z.append(zt); S += zt
Z1 = np.array(z)[:, None]
R1, g1 = ftl_quadratic(Z1)
print(f"[FTL quad, adversarial] regret = {R1:8.2f} <= 4L^2(1+ln T) = {4*L**2*(1+np.log(T2)):.2f}")
assert R1 <= 4 * L**2 * (1 + np.log(T2))

# exact per-round identity: gap_t = 1/2 ||w_t - z_t||^2 (2t-1)/t^2  <= ||w_t - z_t||^2 / t  <= 4L^2/t
t_idx = np.arange(1, T + 1)
assert np.all(gaps <= 4 * L**2 / t_idx + 1e-12)
print("[FTL quad] per-round gap <= 4L^2/t holds for every t")

# ---------- 2. BTL claim: sum_t f_t(w_{t+1}^FTL) <= sum_t f_t(u) for every u ----------
cum_btl = 0.0; S = np.zeros(d)
for t in range(1, T + 1):
    S = S + Z[t - 1]; w_btl = S / t
    cum_btl += 0.5 * np.sum((w_btl - Z[t - 1]) ** 2)
best_fixed = 0.5 * np.sum((Z - Z.mean(0)) ** 2)
print(f"[BTL claim] BTL loss = {cum_btl:.2f} <= best fixed loss = {best_fixed:.2f}")
assert cum_btl <= best_fixed + 1e-9

# ---------- 3. Linear losses in 1-D on S=[-1,1]: FTL has linear regret ----------
def ftl_linear_game(T):
    """Adversary of the lecture: z_1 = 0.5, then z_t = sign of FTL's choice (pushes FTL's loss to +1)."""
    s = 0.0; zs = []; rows = []
    loss_ftl = loss_btl = 0.0
    for t in range(1, T + 1):
        w_ftl = 0.0 if t == 1 else (-1.0 if s > 0 else (1.0 if s < 0 else 0.0))   # argmin_w s*w on [-1,1]
        z = 0.5 if t == 1 else (w_ftl if w_ftl != 0 else 1.0)
        s += z; zs.append(z)
        w_btl = -1.0 if s > 0 else (1.0 if s < 0 else 0.0)
        loss_ftl += w_ftl * z; loss_btl += w_btl * z
        if t <= 4: rows.append((t, w_ftl, z, w_btl, w_ftl * z, w_btl * z))
    best_fixed = -abs(sum(zs))                          # min_{u in [-1,1]} u * sum z
    return rows, loss_ftl, loss_btl, loss_ftl - best_fixed

rows, lf, lb, reg = ftl_linear_game(1000)
print("[FTL linear] round | w_FTL | z_t | w_BTL | loss_FTL | loss_BTL")
for r in rows: print("            ", *[f"{x:6.2f}" for x in r])
print(f"[FTL linear] T=1000: FTL loss = {lf:.1f}, BTL loss = {lb:.1f}, gap = {lf-lb:.1f} (lecture: 2T-1.5 = {2*1000-1.5}), regret = {reg:.1f}")
assert abs((lf - lb) - (2 * 1000 - 1.5)) < 1e-9 and reg > 0.9 * 1000

# ---------- 4. Hedge as a special case of the general protocol: f_t(p) = <p, l_t> ----------
ell = rng.random(4); p = np.array([.1, .2, .3, .4])
assert np.isclose(p @ ell, sum(p[i] * ell[i] for i in range(4)))
print("[Hedge as special case] f_t(p) = <p, l_t> is linear on the simplex: OK")
print("All checks passed.")

```

**Expected output**

```text
[FTL quad, random]      regret =     5.98 <= 4L^2(1+ln T) = 152.28
[FTL quad, adversarial] regret =    22.04 <= 4L^2(1+ln T) = 152.28
[FTL quad] per-round gap <= 4L^2/t holds for every t
[BTL claim] BTL loss = 3331.50 <= best fixed loss = 3336.76
[FTL linear] round | w_FTL | z_t | w_BTL | loss_FTL | loss_BTL
               1.00   0.00   0.50  -1.00   0.00  -0.50
               2.00  -1.00  -1.00   1.00   1.00  -1.00
               3.00   1.00   1.00  -1.00   1.00  -1.00
               4.00  -1.00  -1.00   1.00   1.00  -1.00
[FTL linear] T=1000: FTL loss = 999.0, BTL loss = -999.5, gap = 1998.5 (lecture: 2T-1.5 = 1998.5), regret = 999.5
[Hedge as special case] f_t(p) = <p, l_t> is linear on the simplex: OK
All checks passed.
```

---
*End of Week 5 notes.*
