# Sequential Decision Making — Weeks 1–4
## Detailed lecture notes, mathematical summary, formula sheet, and quiz preparation

> **Scope.** These notes follow the uploaded lecture transcript (`L2-23.txt`) and the five-page handwritten Hedge slide (`week-4-1.pdf`). The lecture sequence moves from learning with expert advice to online learning over a simplex, Hedge, and a combinatorial assignment problem. The transcript also introduces the broader settings of bandits and reinforcement learning.
>
> **Notation convention.** The number of rounds is $T$, the number of experts/actions is $d$, and the learning rate is $\eta>0$. Superscript $t$ denotes a round; subscript $i$ denotes an expert, action, vehicle, or route as appropriate.

---

# 1. The big picture: sequential decision making

In a conventional batch-learning problem, a learner is given a dataset and computes a model from that dataset. In a sequential decision-making problem, decisions are made repeatedly:

1. The learner observes the information currently available.
2. The learner chooses an action.
3. The environment responds.
4. The learner receives feedback.
5. The learner uses that feedback to make the next decision.

The important feature is that the learner must act before seeing the feedback for the current round.

## 1.1 Three broad settings

The professor distinguishes settings by the feedback available to the learner.

| Setting | Feedback after an action | Key characteristic |
|---|---|---|
| **Online learning** | Losses/outcomes for all actions are revealed | Full information |
| **Multi-armed bandits** | Only the loss/reward of the selected action is revealed | Partial information |
| **Reinforcement learning (RL)** | Feedback is partial and actions can affect the state and future possibilities | Partial information with state changes |

### Online learning

Suppose there are $d$ available actions. In round $t$, the environment generates a loss vector

$$
\boldsymbol{\ell}^{\,t}=
(\ell_1^t,\ell_2^t,\ldots,\ell_d^t).
$$

After the learner acts, it sees the entire vector. It therefore knows not only how its chosen action performed, but also how every alternative action would have performed in that round.

### Multi-armed bandits

The learner chooses one action, but observes only the reward or loss of that selected action. It does not directly observe the outcomes of the alternatives.

This creates an exploration problem: the learner must decide whether to use an action that currently appears good or try an action whose quality is still uncertain.

### Reinforcement learning

The learner's action can influence the next state. Consequently, a decision has two effects:

- an immediate reward or loss;
- a possible effect on future states and future rewards.

**Quiz distinction:** Full information versus partial information is the first classification to make. The additional state-transition aspect distinguishes the RL setting described in the lecture.

---

# 2. Learning with expert advice

## 2.1 Problem setup

Assume there are $d$ experts. Each expert gives advice in every round.

For the stock-market illustration, an expert predicts either:

$$
x_i^t\in\{0,1\},
$$

where $1$ means invest and $0$ means do not invest.

The learner also makes a prediction:

$$
\widehat y_t\in\{0,1\}.
$$

After the learner predicts, the actual outcome $y_t\in\{0,1\}$ is revealed.

The mistake indicator is

$$
\mathbf{1}\{\widehat y_t\ne y_t\}=
\begin{cases}
1,&\widehat y_t\ne y_t,\\
0,&\widehat y_t=y_t.
\end{cases}
$$

The cumulative number of mistakes made by the learner is

$$
M_T=
\sum_{t=1}^{T}
\mathbf{1}\{\widehat y_t\ne y_t\}.
$$

The cumulative mistakes of expert $i$ are

$$
M_{i,T}=
\sum_{t=1}^{T}
\mathbf{1}\{x_i^t\ne y_t\}.
$$

The learner wants to make as few mistakes as possible.

## 2.2 Full-information feedback

After $y_t$ is revealed, the learner can determine which experts were correct and which were wrong.

This is crucial. The learner is not restricted to learning whether its own prediction was correct. It receives enough information to evaluate every expert's prediction.

## 2.3 The perfect-expert assumption

The first simplified model assumes that at least one expert is perfect:

$$
\exists i^\star\in\{1,\ldots,d\}
\quad\text{such that}\quad
x_{i^\star}^t=y_t
\quad\forall t.
$$

The identity of this perfect expert is unknown.

The central idea is therefore:

> If we can identify the perfect expert, we can follow that expert and make no further mistakes.

This motivates the Majority Algorithm.

---

# 3. Majority Algorithm

## 3.1 Algorithm

Maintain a candidate set of experts:

$$
C_1=\{1,2,\ldots,d\}.
$$

At round $t$:

1. Receive predictions from the experts in $C_t$.
2. Predict the majority decision among those experts.
3. Observe the actual outcome $y_t$.
4. Remove all experts who made a mistake.

Thus,

$$
C_{t+1}=
\{i\in C_t:x_i^t=y_t\}.
$$

The algorithm always retains the perfect expert, because that expert never makes a mistake.

## 3.2 Why does it work?

Suppose the algorithm makes a mistake in round $t$.

It predicted the majority decision, but the actual outcome was the opposite. Therefore, the majority of the current candidate experts must have been wrong.

The algorithm removes that majority.

At least half the experts are eliminated whenever the algorithm makes a mistake:

$$
|C_{t+1}|\le \frac{|C_t|}{2}.
$$

After $m$ mistakes,

$$
|C_{t+1}|
\le
\frac{d}{2^m}.
$$

Since the perfect expert must remain, the candidate set cannot become empty. In particular, once it contains only one expert, the identity of the perfect expert is known.

Therefore,

$$
\frac{d}{2^m}\ge 1
$$

is necessary for the worst-case sequence of eliminations, giving

$$
m\le \log_2 d.
$$

Since the number of mistakes is an integer,

$$
\boxed{M_T\le \lceil\log_2 d\rceil.}
$$

The exact tie-breaking convention should be specified in a formal implementation. The bound relies on eliminating at least half the candidates after a mistake.

## 3.3 Is Majority optimal?

The lecture also gives a lower-bound argument.

Imagine an adversary that controls the experts' predictions and the outcomes, subject to the condition that at least one perfect expert must remain.

At each round, the adversary can:

1. Split the currently viable experts approximately equally between predictions $0$ and $1$.
2. Observe the learner's prediction.
3. Choose the opposite outcome, forcing a learner mistake.
4. Retain the half of the experts that predicted that outcome.

The number of viable experts is reduced by at most a factor of two per forced mistake.

The adversary can continue for approximately $\log_2 d$ rounds while preserving at least one perfect expert.

Thus the lower bound is of logarithmic order, matching the Majority Algorithm's upper bound.

**Conclusion:** Under the perfect-expert assumption, the Majority Algorithm is optimal in worst-case mistake complexity, up to the precise integer/tie convention.

## 3.4 Limitation

The perfect-expert assumption is unrealistic in many applications. There may be no expert who is always correct.

This motivates a more practical question:

> Can the learner perform nearly as well as the best expert, even if every expert makes mistakes?

That leads to regret and Weighted Majority.

---

# 4. Regret: the central performance measure

Mistake count is not always the right measure. If no expert is perfect, we need to compare the learner against the best available expert.

The regret after $T$ rounds is

$$
\boxed{
R_T=
M_T-\min_{i\in[d]}M_{i,T}.
}
$$

In words:

$$
\text{Regret}=\text{Learner's cumulative mistakes}-
\text{Best expert's cumulative mistakes in hindsight}.
$$

The best expert is selected after observing all $T$ rounds. It need not be the same expert that appeared best at an earlier stopping time.

## 4.1 Why is it called regret?

Suppose the learner makes 400 mistakes, while the best expert makes 40 mistakes.

Then

$$
R_T=400-40=360.
$$

The regret measures the additional mistakes that could have been avoided if the learner had known in advance which expert would perform best over the entire sequence.

## 4.2 Sublinear regret

An algorithm has sublinear regret if

$$
\boxed{R_T=o(T).}
$$

Equivalently,

$$
\lim_{T\to\infty}\frac{R_T}{T}=0.
$$

This does **not** mean that total regret must decrease. It means that average regret per round approaches zero.

For example, if

$$
R_T=O(\sqrt T),
$$

then

$$
\frac{R_T}{T}=O\left(\frac{1}{\sqrt T}\right)\longrightarrow 0.
$$

This is also called no-regret learning.

### Important distinction

- **Total regret:** $R_T$
- **Average regret:** $R_T/T$
- **Sublinear regret:** total regret grows slower than $T$
- **No-regret property:** average regret tends to zero

---

# 5. Weighted Majority Algorithm

## 5.1 Motivation

The Majority Algorithm treats every expert equally and assumes a perfect expert exists.

Weighted Majority relaxes the perfect-expert assumption.

Instead of maintaining only a set of candidate experts, it maintains a positive weight for every expert:

$$
w_i^t>0.
$$

The weight represents the learner's current confidence or relative preference for expert $i$.

Initially,

$$
w_i^1=1
\qquad \forall i.
$$

## 5.2 Weight update

When an expert makes a mistake, reduce its weight.

A common update is

$$
w_i^{t+1}=
\begin{cases}
(1-\epsilon)w_i^t,
&\text{if expert }i\text{ is wrong},\\
w_i^t,
&\text{if expert }i\text{ is correct},
\end{cases}
$$

where $0<\epsilon<1$.

An expert that repeatedly makes mistakes loses influence exponentially.

For example, after $m$ mistakes, an expert's weight is

$$
w_i=(1-\epsilon)^m w_i^1.
$$

## 5.3 Prediction

The deterministic Weighted Majority Algorithm predicts according to the larger total weight of the experts supporting each decision.

For binary predictions, define

$$
W_1^t=\sum_{i:x_i^t=1}w_i^t,
\qquad
W_0^t=\sum_{i:x_i^t=0}w_i^t.
$$

Predict

$$
\widehat y_t=
\begin{cases}
1,&W_1^t\ge W_0^t,\\
0,&W_1^t<W_0^t.
\end{cases}
$$

The tie-breaking rule can be chosen consistently.

## 5.4 Why use weights?

The algorithm does not need to identify a perfect expert.

Instead:

- experts that perform well retain larger weights;
- experts that perform poorly lose weight;
- the prediction is influenced more strongly by experts with larger weights.

This is a soft alternative to the elimination mechanism of Majority.

## 5.5 Limitation: an adaptive adversary

A deterministic algorithm reveals its decision completely.

An adversary that can observe the learner's current prediction before choosing the outcome can always choose the opposite outcome. This can force a mistake in every round.

The lecture establishes a negative result: under a sufficiently powerful adaptive adversary, no deterministic algorithm can guarantee sublinear regret in general.

The issue is not merely a poor choice of weight-update rule. It is the power granted to the adversary.

---

# 6. Oblivious adversary and randomization

## 6.1 Adaptive versus oblivious adversary

An adaptive adversary may choose the outcome in round $t$ after observing the learner's actual decision in that round.

An oblivious adversary:

1. Knows the learner's algorithm, including its probabilities.
2. Chooses the entire outcome sequence before the game begins.
3. Does not observe the actual random outcomes generated by the learner during play.

The distinction is subtle but fundamental.

An oblivious adversary may know that the learner predicts $1$ with probability $0.7$. It does not know whether the learner's random draw in the current round actually produces $1$ or $0$.

## 6.2 Why randomization helps

If the learner is deterministic, publishing the algorithm reveals its future decisions to an adversary that knows the outcome sequence it wants to use.

If the learner is randomized, publishing the algorithm reveals the probability distribution, but not the realization of the random bits.

The adversary must commit to outcomes without knowing the learner's actual random decisions.

**Quiz distinction:**

| Adversary | Knows algorithm? | Knows actual random realization before choosing outcomes? |
|---|---:|---:|
| Adaptive | Yes | Can react to the current decision |
| Oblivious | Yes | No |

Randomization is useful because it hides the realization, not because it hides the algorithm or its probabilities.

---

# 7. Expected mistakes and randomized Weighted Majority

Once the learner uses randomness, its mistake count becomes a random variable.

A useful performance measure is its expected number of mistakes:

$$
\mathbb E[M_T]=
\mathbb E\left[
\sum_{t=1}^{T}
\mathbf 1\{\widehat y_t\ne y_t\}
\right].
$$

By linearity of expectation,

$$
\boxed{
\mathbb E[M_T]=
\sum_{t=1}^{T}
\Pr(\widehat y_t\ne y_t).
}
$$

The expectation is over the learner's random choices (and over the environment's randomness as well, if the environment is randomized).

## 7.1 Randomized prediction using weights

The learner can choose its binary prediction according to the fraction of total weight supporting each prediction.

Define

$$
p_t(1)=
\frac{\sum_{i:x_i^t=1}w_i^t}
{\sum_{i=1}^{d}w_i^t},
$$

and

$$
p_t(0)=1-p_t(1).
$$

The learner samples its prediction from this distribution.

For a fixed outcome $y_t$, its probability of making a mistake is

$$
\Pr(\widehat y_t\ne y_t).
$$

For example, if the true outcome is $1$, the probability of a mistake is $p_t(0)$.

This expected mistake can be represented as a weighted average of the experts' individual mistake indicators.

## 7.2 Randomized Weighted Majority update

A common multiplicative update is

$$
w_i^{t+1}=
w_i^t(1-\epsilon)^{\ell_i^t},
$$

where

$$
\ell_i^t=
\mathbf 1\{x_i^t\ne y_t\}.
$$

Thus:

- if expert $i$ is correct, $\ell_i^t=0$, so its weight is unchanged;
- if expert $i$ is wrong, $\ell_i^t=1$, so its weight is multiplied by $1-\epsilon$.

The lecture's main conceptual point is that the algorithm randomizes its prediction while using the full-information feedback to update all expert weights.

## 7.3 Regret guarantee

The weighted-majority analysis gives a bound of the general form

$$
\mathbb E[M_T]
\le
(1+\epsilon)M^\star_T
+
O\left(\frac{\ln d}{\epsilon}\right),
$$

where

$$
M^\star_T=\min_i M_{i,T}.
$$

Consequently,

$$
\mathbb E[R_T]
\le
\epsilon M^\star_T
+
O\left(\frac{\ln d}{\epsilon}\right).
$$

Since $M^\star_T\le T$, a suitable choice of $\epsilon$ gives a regret bound of order

$$
\boxed{
\mathbb E[R_T]=O(\sqrt{T\ln d}).
}
$$

The precise constants depend on the chosen update convention and the version of the weighted-majority bound being used.

The lecture also discusses a lower-bound argument showing that square-root dependence on $T$ is unavoidable in the relevant adversarial setting, up to logarithmic factors.

---

# 8. From experts to a linear game on a simplex

The next conceptual step is to generalize the expert problem.

Instead of making a binary prediction based on experts, suppose the learner has $d$ possible actions.

The learner maintains a probability vector

$$
p^t=(p_1^t,\ldots,p_d^t).
$$

It must satisfy

$$
p_i^t\ge 0,
\qquad
\sum_{i=1}^{d}p_i^t=1.
$$

The set of all such vectors is the probability simplex:

$$
\boxed{
\Delta_d=
\left\{
p\in\mathbb R^d:
p_i\ge0,\;
\sum_{i=1}^{d}p_i=1
\right\}.
}
$$

The environment returns a loss vector

$$
\ell^t=(\ell_1^t,\ldots,\ell_d^t).
$$

The learner's loss is the dot product

$$
\boxed{
\langle p^t,\ell^t\rangle=
\sum_{i=1}^{d}p_i^t\ell_i^t.
}
$$

This is a linear game on the simplex.

## 8.1 Two interpretations of $p^t$

There are two equivalent ways to interpret the probability vector.

### Interpretation A: play the mixture

The learner directly plays the point $p^t$ in the simplex and suffers

$$
\langle p^t,\ell^t\rangle.
$$

This is appropriate when a mixture is physically meaningful, such as distributing a budget across several investments.

### Interpretation B: sample one action

The learner samples action $i$ with probability $p_i^t$, and plays that action.

The actual loss is random, but its expected value is

$$
\mathbb E[\ell_{\text{chosen}}^t]=
\sum_{i=1}^{d}p_i^t\ell_i^t=
\langle p^t,\ell^t\rangle.
$$

This is appropriate when the learner must make one indivisible choice.

**Important:** The expected loss is the same in both interpretations. The realized loss need not be the same.

## 8.2 Regret on the simplex

The learner is compared against the best fixed point in the simplex:

$$
\boxed{
R_T=
\sum_{t=1}^{T}\langle p^t,\ell^t\rangle-
\min_{p\in\Delta_d}
\sum_{t=1}^{T}\langle p,\ell^t\rangle.
}
$$

Since $p$ is fixed across all rounds in the hindsight comparison, linearity gives

$$
\min_{p\in\Delta_d}
\sum_{t=1}^{T}\langle p,\ell^t\rangle=
\min_{p\in\Delta_d}
\left\langle p,\sum_{t=1}^{T}\ell^t\right\rangle.
$$

A linear function over a simplex achieves its minimum at a vertex. Therefore,

$$
\boxed{
\min_{p\in\Delta_d}
\sum_{t=1}^{T}\langle p,\ell^t\rangle=
\min_{i\in[d]}\sum_{t=1}^{T}\ell_i^t.
}
$$

This is an important result: even though the comparator is allowed to be any point in the simplex, the best fixed comparator can be chosen as one of the $d$ vertices.

The vertices correspond to the pure actions:

$$
e_1=(1,0,\ldots,0),\quad
e_2=(0,1,\ldots,0),\quad\ldots
$$

---

# 9. Hedge algorithm

Hedge generalizes multiplicative weight updates to arbitrary nonnegative losses.

## 9.1 Algorithm

Initialize all weights to one:

$$
w_i^1=1,\qquad i=1,\ldots,d.
$$

At round $t$:

### Step 1: Normalize the weights

Define

$$
\Phi^t=\sum_{i=1}^{d}w_i^t.
$$

Then

$$
\boxed{
p_i^t=\frac{w_i^t}{\Phi^t}.
}
$$

Because all weights are positive,

$$
p_i^t>0,\qquad \sum_i p_i^t=1.
$$

Thus $p^t\in\Delta_d$.

### Step 2: Play

Play $p^t$, or sample a pure action according to $p^t$, depending on the application.

### Step 3: Receive the loss vector

The environment reveals

$$
\ell^t=(\ell_1^t,\ldots,\ell_d^t).
$$

### Step 4: Update weights

$$
\boxed{
w_i^{t+1}=
w_i^t\exp(-\eta\ell_i^t).
}
$$

Here $\eta>0$ is the learning rate.

## 9.2 Interpretation of the update

If losses are nonnegative:

- a larger loss produces a smaller multiplicative factor;
- a smaller loss produces a factor closer to one;
- an action with a small loss retains more relative weight.

The update is multiplicative, not additive.

The exponential ensures that weights remain strictly positive, provided they start positive and losses are finite.

## 9.3 Connection to Weighted Majority

For binary mistake losses,

$$
\ell_i^t\in\{0,1\}.
$$

Hedge gives

$$
w_i^{t+1}=
\begin{cases}
w_i^t,&\ell_i^t=0,\\
w_i^te^{-\eta},&\ell_i^t=1.
\end{cases}
$$

This is the same type of multiplicative reduction used by Weighted Majority, with the mistake penalty factor

$$
\beta=e^{-\eta}.
$$

Thus Hedge is a more general exponential-weight algorithm.

---

# 10. Hedge regret analysis — the professor's slide

This is the central derivation on the uploaded slide. It is worth understanding each inequality rather than memorizing only the final bound.

## 10.1 Assumptions

Assume that every loss is bounded:

$$
0\le\ell_i^t\le B.
$$

The slide uses two exponential inequalities:

$$
\boxed{e^x\ge 1+x\qquad\text{for all }x\in\mathbb R}
\tag{A}
$$

and

$$
\boxed{
e^x\le 1+x+\frac{x^2}{2}
\qquad\text{for }x\le0.
}
\tag{B}
$$

The second inequality is used with $x=-\eta\ell_i^t$, which is nonpositive under the loss assumption.

## 10.2 Track the total weight

Recall

$$
\Phi^t=\sum_{i=1}^{d}w_i^t.
$$

Using the Hedge update,

$$
\begin{aligned}
\frac{\Phi^{t+1}}{\Phi^t}
&=
\frac{\sum_i w_i^{t+1}}{\Phi^t}\\
&=
\frac{\sum_i w_i^t e^{-\eta\ell_i^t}}{\Phi^t}\\
&=
\sum_i p_i^t e^{-\eta\ell_i^t}.
\end{aligned}
$$

This ratio is the key quantity in the proof.

## 10.3 Apply the quadratic exponential bound

Using inequality (B),

$$
e^{-\eta\ell_i^t}
\le
1-\eta\ell_i^t
+\frac{\eta^2(\ell_i^t)^2}{2}.
$$

Substitute into the weight ratio:

$$
\begin{aligned}
\frac{\Phi^{t+1}}{\Phi^t}
&\le
\sum_i p_i^t
\left(
1-\eta\ell_i^t
+\frac{\eta^2(\ell_i^t)^2}{2}
\right)\\
&=
1-\eta\langle p^t,\ell^t\rangle
+\frac{\eta^2}{2}
\left\langle p^t,(\ell^t)^2\right\rangle.
\end{aligned}
$$

Here $(\ell^t)^2$ means componentwise squaring:

$$
(\ell^t)^2
=
((\ell_1^t)^2,\ldots,(\ell_d^t)^2).
$$

Now use inequality (A), in the form $1+z\le e^z$:

$$
\frac{\Phi^{t+1}}{\Phi^t}
\le
\exp\left(
-\eta\langle p^t,\ell^t\rangle
+
\frac{\eta^2}{2}
\left\langle p^t,(\ell^t)^2\right\rangle
\right).
$$

Taking logarithms,

$$
\boxed{
\ln\frac{\Phi^{t+1}}{\Phi^t}
\le
-\eta\langle p^t,\ell^t\rangle
+
\frac{\eta^2}{2}
\left\langle p^t,(\ell^t)^2\right\rangle.
}
$$

## 10.4 Sum over all rounds

Summing from $t=1$ to $T$,

$$
\sum_{t=1}^{T}
\ln\frac{\Phi^{t+1}}{\Phi^t}
\le
-\eta\sum_{t=1}^{T}\langle p^t,\ell^t\rangle
+
\frac{\eta^2}{2}
\sum_{t=1}^{T}
\left\langle p^t,(\ell^t)^2\right\rangle.
$$

The left side telescopes:

$$
\ln\Phi^{T+1}-\ln\Phi^1
\le
-\eta\sum_{t=1}^{T}\langle p^t,\ell^t\rangle
+
\frac{\eta^2}{2}
\sum_{t=1}^{T}
\left\langle p^t,(\ell^t)^2\right\rangle.
$$

Initially all weights are one, so

$$
\Phi^1=d.
$$

Therefore,

$$
\boxed{
\ln\Phi^{T+1}
\le
\ln d
-\eta\sum_{t=1}^{T}\langle p^t,\ell^t\rangle
+
\frac{\eta^2}{2}
\sum_{t=1}^{T}
\left\langle p^t,(\ell^t)^2\right\rangle.
}
\tag{1}
$$

## 10.5 Lower-bound the final total weight

For any fixed action $j$,

$$
\Phi^{T+1}\ge w_j^{T+1}.
$$

Repeatedly applying the update gives

$$
w_j^{T+1}=
\exp\left(
-\eta\sum_{t=1}^{T}\ell_j^t
\right),
$$

because $w_j^1=1$.

Hence,

$$
\ln\Phi^{T+1}
\ge-\eta\sum_{t=1}^{T}\ell_j^t.
$$

Choose $j$ to be the best fixed action in hindsight:

$$
\boxed{
\ln\Phi^{T+1}
\ge-\eta\min_j\sum_{t=1}^{T}\ell_j^t.
}
\tag{2}
$$

## 10.6 Combine the bounds

Combining (1) and (2),

$$
-\eta\min_j\sum_t\ell_j^t
\le
\ln d
-\eta\sum_t\langle p^t,\ell^t\rangle
+
\frac{\eta^2}{2}
\sum_t\langle p^t,(\ell^t)^2\rangle.
$$

Rearranging and dividing by $\eta>0$,

$$
\boxed{
R_T
\le
\frac{\ln d}{\eta}
+
\frac{\eta}{2}
\sum_{t=1}^{T}
\left\langle p^t,(\ell^t)^2\right\rangle.
}
$$

This is the more informative intermediate bound.

## 10.7 Use the bounded-loss assumption

Since

$$
0\le\ell_i^t\le B,
$$

we have

$$
(\ell_i^t)^2\le B^2.
$$

Since $p^t$ is a probability vector,

$$
\left\langle p^t,(\ell^t)^2\right\rangle=
\sum_i p_i^t(\ell_i^t)^2
\le
B^2\sum_i p_i^t=
B^2.
$$

Therefore,

$$
\boxed{
R_T
\le
\frac{\ln d}{\eta}
+
\frac{\eta TB^2}{2}.
}
$$

## 10.8 Choose the optimal learning rate

We want to minimize

$$
f(\eta)=\frac{\ln d}{\eta}+\frac{\eta TB^2}{2}.
$$

Differentiate:

$$
f'(\eta)=
-\frac{\ln d}{\eta^2}
+\frac{TB^2}{2}.
$$

Set $f'(\eta)=0$:

$$
\frac{\ln d}{\eta^2}=
\frac{TB^2}{2}.
$$

Thus,

$$
\eta^2=\frac{2\ln d}{TB^2},
$$

and

$$
\boxed{
\eta^\star=
\frac{1}{B}
\sqrt{\frac{2\ln d}{T}}.
}
$$

Substituting this value,

$$
\boxed{
R_T\le B\sqrt{2T\ln d}.
}
$$

For losses in $[0,1]$, set $B=1$:

$$
\boxed{
R_T\le\sqrt{2T\ln d}.
}
$$

Hence,

$$
\boxed{R_T=O(\sqrt{T\ln d}).}
$$

Dividing by $T$,

$$
\frac{R_T}{T}=
O\left(\sqrt{\frac{\ln d}{T}}\right)
\longrightarrow0
$$

for fixed $d$.

### What the bound tells us

- More rounds increase the regret bound as $\sqrt T$.
- More actions increase it as $\sqrt{\ln d}$.
- The average regret tends to zero.
- The learning rate should decrease as the time horizon $T$ increases.

---

# 11. Combinatorial decision spaces: the assignment problem

Hedge is attractive because its regret bound depends logarithmically on the number of actions.

However, the algorithm must maintain a weight for every action.

If the number of actions is factorial or exponential, this becomes computationally infeasible.

## 11.1 Assignment problem

Suppose a transport company has:

- $n$ vehicles;
- $n$ routes.

Every vehicle must be assigned to exactly one route, and every route must receive exactly one vehicle.

An assignment is a permutation.

The number of possible assignments is

$$
\boxed{n!.}
$$

For $n=4$, there are

$$
4!=24
$$

possible assignments.

For $n=10$, there are

$$
10!=3,628,800
$$

possible assignments.

Running ordinary Hedge over all these actions requires maintaining $n!$ weights.

This is the computational difficulty.

## 11.2 Regret for the assignment problem

Let $M^t$ denote the assignment chosen in round $t$, and let $L^t$ be the loss matrix.

The loss of an assignment is the sum of the losses of the selected vehicle-route pairs.

The learner wants to compare its cumulative loss against the best fixed assignment in hindsight:

$$
R_T=
\sum_{t=1}^{T}\langle M^t,L^t\rangle-
\min_{M\in\mathcal P_n}
\sum_{t=1}^{T}\langle M,L^t\rangle,
$$

where $\mathcal P_n$ is the set of all $n\times n$ permutation matrices.

---

# 12. Permutation matrices

An assignment can be represented by an $n\times n$ matrix $M$:

$$
M_{ij}=
\begin{cases}
1,&\text{vehicle }i\text{ is assigned to route }j,\\
0,&\text{otherwise}.
\end{cases}
$$

Every row contains exactly one $1$, because each vehicle gets one route.

Every column contains exactly one $1$, because each route gets one vehicle.

Therefore,

$$
\boxed{
\sum_{j=1}^{n}M_{ij}=1
\quad\forall i
}
$$

and

$$
\boxed{
\sum_{i=1}^{n}M_{ij}=1
\quad\forall j.
}
$$

Such a matrix is called a permutation matrix.

There are $n!$ distinct permutation matrices of size $n\times n$.

## 12.1 Linear loss

Let $L^t$ be the loss matrix, where $L_{ij}^t$ is the loss of assigning vehicle $i$ to route $j$.

The assignment loss is

$$
\boxed{
\langle M,L^t\rangle=
\sum_{i=1}^{n}\sum_{j=1}^{n}M_{ij}L_{ij}^t.
}
$$

Because $M$ has a $1$ only at selected assignments, this expression adds the losses of the chosen vehicle-route pairs.

This is a linear loss in the matrix representation.

---

# 13. Doubly stochastic matrices and the Birkhoff polytope

A doubly stochastic matrix $P$ satisfies

$$
P_{ij}\ge0,
$$

$$
\sum_{j=1}^{n}P_{ij}=1
\quad\forall i,
$$

and

$$
\sum_{i=1}^{n}P_{ij}=1
\quad\forall j.
$$

In words, every entry is nonnegative, and every row and every column sums to one.

A permutation matrix is a special case of a doubly stochastic matrix.

## 13.1 Convex combinations

A convex combination of matrices $M_1,\ldots,M_k$ is

$$
P=\sum_{r=1}^{k}\lambda_rM_r,
$$

where

$$
\lambda_r\ge0,
\qquad
\sum_{r=1}^{k}\lambda_r=1.
$$

A convex combination of permutation matrices is doubly stochastic.

The Birkhoff–von Neumann theorem gives the converse:

$$
\boxed{
\text{Every doubly stochastic matrix is a convex combination of permutation matrices.}
}
$$

Thus the set of doubly stochastic matrices is exactly the convex hull of all permutation matrices. This set is called the Birkhoff polytope.

## 13.2 Why is this useful?

Instead of maintaining a probability distribution over $n!$ assignments, we can maintain a single $n\times n$ doubly stochastic matrix.

This is a representation using $n^2$ entries rather than $n!$ probabilities.

However, when the actual decision must be one assignment, we still need to sample a permutation matrix from a decomposition of the doubly stochastic matrix.

---

# 14. Carathéodory's theorem

Carathéodory's theorem states:

If a point lies in the convex hull of a set in $\mathbb R^m$, then it can be represented as a convex combination of at most $m+1$ points from that set.

An $n\times n$ matrix can be viewed as a vector in

$$
\mathbb R^{n^2}.
$$

Therefore, any doubly stochastic matrix can be represented as a convex combination of at most

$$
\boxed{n^2+1}
$$

permutation matrices by the general Carathéodory bound.

The lecture then develops a constructive matching-based decomposition that obtains at most $n^2$ permutation matrices.

**Do not confuse the two bounds:**

- Carathéodory's general guarantee: $n^2+1$.
- The lecture's iterative matching decomposition: at most $n^2$.

---

# 15. Decomposing a doubly stochastic matrix

The professor constructs a decomposition algorithm using a weighted bipartite graph.

## 15.1 Convert the matrix into a bipartite graph

Given a doubly stochastic matrix $A$:

- Create $n$ vertices on the left, representing vehicles.
- Create $n$ vertices on the right, representing routes.
- Add an edge from left vertex $i$ to right vertex $j$ whenever $A_{ij}>0$.
- Give that edge weight $A_{ij}$.

Thus,

$$
(i,j)\text{ is an edge}
\iff A_{ij}>0.
$$

## 15.2 Find a perfect matching

A matching is a collection of edges that do not share endpoints.

A perfect matching covers every vertex on both sides.

Since there are $n$ vertices on each side, a perfect matching contains exactly $n$ edges.

Each perfect matching corresponds to one permutation matrix.

## 15.3 Find the minimum edge weight

Suppose the selected matching corresponds to permutation matrix $M$.

Find the minimum weight among the selected edges:

$$
\boxed{
\lambda=
\min_{(i,j):M_{ij}=1}A_{ij}.
}
$$

## 15.4 Subtract the matching

Update the residual matrix:

$$
\boxed{
A\leftarrow A-\lambda M.
}
$$

This does three important things:

1. It removes a positive amount from every selected matching edge.
2. It makes at least one matrix entry zero, because $\lambda$ was the minimum selected edge weight.
3. It reduces every row sum and every column sum by the same amount $\lambda$.

If the original row and column sums were $c$, the new row and column sums are

$$
c-\lambda.
$$

The residual matrix need not be doubly stochastic with row and column sums equal to one, but it remains a scaled doubly stochastic matrix.

## 15.5 Repeat

Repeat:

1. Construct the graph from positive entries.
2. Find a perfect matching.
3. Form its permutation matrix.
4. Find the minimum selected edge weight.
5. Subtract that weighted permutation matrix.

Stop when the residual matrix becomes zero.

## 15.6 Why does a perfect matching continue to exist?

At every stage, the residual matrix has nonnegative entries and equal row and column sums.

If the common sum is $c>0$, then

$$
\frac{1}{c}A
$$

is doubly stochastic.

By the Birkhoff–von Neumann theorem, it is a convex combination of permutation matrices. Hence at least one permutation matrix must have its $1$-entries only at positive entries of the residual matrix.

That permutation matrix corresponds to a perfect matching in the current bipartite graph.

Therefore, the matching required by the algorithm exists at every nonzero stage.

## 15.7 Why does the algorithm terminate in polynomial time?

At each iteration, at least one positive matrix entry becomes zero.

There are only $n^2$ entries.

Therefore, the algorithm performs at most

$$
\boxed{n^2}
$$

iterations.

Each iteration requires finding a perfect matching in a bipartite graph, which can be done in polynomial time.

Thus the complete decomposition algorithm runs in polynomial time.

## 15.8 Final decomposition

The procedure gives

$$
\boxed{
A=\sum_{r=1}^{k}\lambda_rM_r,
\qquad k\le n^2,
}
$$

where:

- $M_r$ are permutation matrices;
- $\lambda_r\ge0$;
- $\sum_r\lambda_r=1$, if the original matrix was doubly stochastic.

The coefficients form a probability distribution over the selected permutation matrices.

---

# 16. Combinatorial Hedge

We now have the tools needed to extend Hedge to the assignment problem.

## 16.1 What must be maintained?

Instead of a vector of weights over $n!$ assignments, maintain an $n\times n$ positive weight matrix:

$$
W^1_{ij}=1.
$$

The entry $W_{ij}^t$ represents the current relative preference for assigning vehicle $i$ to route $j$.

## 16.2 Convert the weight matrix into a doubly stochastic matrix

The weight matrix does not necessarily have row and column sums equal to one.

We need to transform it into a doubly stochastic matrix $P^t$.

Simple row normalization is insufficient:

- it makes every row sum to one;
- it does not generally make every column sum to one.

The lecture introduces the **Sinkhorn balancing algorithm**.

### Sinkhorn balancing

Repeatedly perform:

1. Normalize rows to sum to one.
2. Normalize columns to sum to one.
3. Repeat until the matrix is sufficiently close to doubly stochastic.

For a strictly positive matrix, this iterative scaling converges under the standard conditions for Sinkhorn scaling.

The resulting matrix provides the desired doubly stochastic representation.

## 16.3 Decompose and sample

Use the matching-based algorithm to decompose

$$
P^t=
\sum_{r=1}^{k_t}\lambda_r^tM_r^t,
\qquad
k_t\le n^2.
$$

Sample one permutation matrix $M_r^t$ with probability $\lambda_r^t$.

This is the actual assignment played by the learner.

The expected loss is

$$
\boxed{
\mathbb E[\text{loss at }t]=
\langle P^t,L^t\rangle.
}
$$

## 16.4 Update the weights

After receiving the loss matrix $L^t$, update each entry:

$$
\boxed{
W_{ij}^{t+1}=
W_{ij}^t\exp(-\eta L_{ij}^t).
}
$$

This is exactly the Hedge update, applied entry by entry to a matrix.

## 16.5 Full algorithm at a glance

For every round $t$:

1. Start with positive weight matrix $W^t$.
2. Apply Sinkhorn balancing to obtain $P^t$.
3. Decompose $P^t$ into at most $n^2$ weighted permutation matrices.
4. Sample one permutation matrix using the decomposition coefficients.
5. Play the corresponding assignment.
6. Receive the loss matrix $L^t$.
7. Update every weight using the exponential multiplicative rule.

The conceptual structure is still Hedge:

$$
\text{weights}
\rightarrow
\text{normalized decision}
\rightarrow
\text{loss}
\rightarrow
\text{weight update}.
$$

The difference is that the weights and normalized decision are now matrices.

---

# 17. Important mathematical clarification: Sinkhorn projection

The lecture motivates Sinkhorn balancing as a matrix analogue of normalizing a weight vector onto the simplex.

There is a useful technical distinction to remember:

- Ordinary vector normalization gives a probability vector.
- Sinkhorn balancing alternately rescales rows and columns to obtain a doubly stochastic matrix.
- In standard mathematical treatments, Sinkhorn scaling is associated with a KL-divergence / entropy-based projection (often called an I-projection), not in general the ordinary Euclidean/Frobenius nearest-point projection.

Thus, if a question asks only for the algorithm taught in the lecture, remember **alternate row and column normalization**. If it asks for the precise optimization metric, do not automatically call it a Frobenius projection.

---

# 18. Formula sheet

| Concept | Formula | Meaning / condition |
|---|---|---|
| Mistake indicator | $\mathbf 1\{\widehat y_t\ne y_t\}$ | 1 for a mistake, 0 otherwise |
| Learner mistakes | $M_T=\sum_{t=1}^T\mathbf 1\{\widehat y_t\ne y_t\}$ | Total mistakes |
| Expert mistakes | $M_{i,T}=\sum_{t=1}^T\mathbf 1\{x_i^t\ne y_t\}$ | Mistakes of expert $i$ |
| Majority bound | $M_T\le\lceil\log_2d\rceil$ | At least one perfect expert |
| Regret | $R_T=M_T-\min_iM_{i,T}$ | Compare with best expert in hindsight |
| Sublinear regret | $R_T=o(T)$ | $R_T/T\to0$ |
| Probability simplex | $\Delta_d=\{p:p_i\ge0,\sum_i p_i=1\}$ | Valid probability vectors |
| Linear loss | $\langle p^t,\ell^t\rangle=\sum_i p_i^t\ell_i^t$ | Loss of a mixture |
| Expected sampled loss | $\mathbb E[\ell_{\rm chosen}^t]=\langle p^t,\ell^t\rangle$ | Sample one action according to $p^t$ |
| Hedge normalization | $p_i^t=w_i^t/\Phi^t$ | $\Phi^t=\sum_iw_i^t$ |
| Hedge update | $w_i^{t+1}=w_i^te^{-\eta\ell_i^t}$ | Multiplicative exponential update |
| Hedge regret | $R_T\le\frac{\ln d}{\eta}+\frac{\eta}{2}\sum_t\langle p^t,(\ell^t)^2\rangle$ | Intermediate bound |
| Bounded-loss Hedge | $R_T\le\frac{\ln d}{\eta}+\frac{\eta TB^2}{2}$ | If $0\le\ell_i^t\le B$ |
| Optimal learning rate | $\eta^\star=\frac1B\sqrt{\frac{2\ln d}{T}}$ | Minimizes the preceding bound |
| Final Hedge bound | $R_T\le B\sqrt{2T\ln d}$ | Under the stated assumptions |
| Assignment count | $n!$ | Number of one-to-one assignments |
| Permutation matrix | One 1 per row and column | Represents one assignment |
| Assignment loss | $\langle M,L\rangle=\sum_{ij}M_{ij}L_{ij}$ | Total selected-pair loss |
| Doubly stochastic matrix | $P_{ij}\ge0,\sum_jP_{ij}=1,\sum_iP_{ij}=1$ | Row and column sums equal one |
| Convex decomposition | $P=\sum_r\lambda_rM_r$ | $\lambda_r\ge0,\sum_r\lambda_r=1$ |
| Carathéodory bound | At most $n^2+1$ matrices | General dimension-$n^2$ guarantee |
| Matching decomposition | At most $n^2$ matrices | Bound from iterative edge elimination |
| Matching coefficient | $\lambda=\min_{(i,j):M_{ij}=1}A_{ij}$ | Largest safe amount to subtract |
| Residual update | $A\leftarrow A-\lambda M$ | Removes at least one positive entry |
| Matrix Hedge update | $W_{ij}^{t+1}=W_{ij}^te^{-\eta L_{ij}^t}$ | Entrywise exponential update |

---

# 19. Inequalities and identities to remember

### Exponential inequalities used in the Hedge proof

$$
\boxed{e^x\ge1+x\quad\forall x\in\mathbb R}
$$

$$
\boxed{e^x\le1+x+\frac{x^2}{2}\quad x\le0}
$$

### Linearity of expectation

$$
\boxed{
\mathbb E\left[\sum_t X_t\right]=
\sum_t\mathbb E[X_t].
}
$$

Independence is not required.

### Linearity of the dot product

$$
\sum_t\langle p,\ell^t\rangle=
\left\langle p,\sum_t\ell^t\right\rangle.
$$

This is valid when the same fixed comparator $p$ is used in every round.

### Minimum of a linear function over a simplex

$$
\boxed{
\min_{p\in\Delta_d}\langle p,z\rangle=
\min_i z_i.
}
$$

A linear objective over a simplex achieves its minimum at a vertex.

### Convex combination

$$
x=\sum_i\lambda_ix_i,
\qquad
\lambda_i\ge0,
\qquad
\sum_i\lambda_i=1.
$$

The coefficients are probabilities when used to sample an action.

---

# 20. Quiz preparation: common traps and tips

## 20.1 Conceptual traps

| Question pattern | Correct reasoning |
|---|---|
| Does Majority work without a perfect expert? | Its logarithmic mistake guarantee relies on at least one perfect expert. |
| Why remove the majority after a mistake? | The majority predicted incorrectly, so the perfect expert must be in the remaining minority. |
| Is the best expert fixed during the game? | The comparator is selected in hindsight after all rounds. |
| Does sublinear regret mean regret decreases? | No. Total regret may grow; its ratio to $T$ tends to zero. |
| What does full information mean? | The learner sees losses/outcomes for all actions, not just its selected action. |
| Does an oblivious adversary know the algorithm? | Yes. It does not know the realized random bits when it commits to the sequence. |
| Why randomize? | To prevent the adversary from reacting to the learner's realized current decision. |
| Does Hedge update additively? | No. It uses multiplicative exponential updates. |
| Can Hedge weights become exactly zero? | Not from a finite exponential update starting with positive weights. |
| Does a probability vector have to be strictly positive? | No. Nonnegative entries summing to one are sufficient. Hedge happens to maintain strictly positive entries under finite losses. |
| Does a doubly stochastic matrix have one 1 per row and column? | No. That is true of a permutation matrix. A doubly stochastic matrix may contain fractional entries. |
| Does representing assignments as matrices reduce the number of assignments? | No. There are still $n!$ assignments. It reduces the dimension of each representation. |
| Why is a perfect matching relevant? | A perfect matching corresponds exactly to a one-to-one assignment and a permutation matrix. |
| Why does subtracting the minimum matching weight help? | It guarantees at least one positive entry becomes zero without making any entry negative. |

## 20.2 Formula-selection tips

- If the question asks for the number of assignments of $n$ vehicles to $n$ routes one-to-one, use $n!$.
- If it asks for the number of entries in the matrix representation, use $n^2$.
- If it asks for the number of permutation matrices in a general convex decomposition, distinguish the Carathéodory bound $n^2+1$ from the lecture's constructive $n^2$ bound.
- If it asks for Hedge's final regret bound, first identify the loss range $B$.
- If losses are in $[0,1]$, substitute $B=1$.
- If asked to choose $\eta$, minimize the two-term expression rather than guessing.
- If a loss vector is supplied, compute a dot product using every component.
- If a matrix is claimed to be doubly stochastic, check both row sums and column sums, as well as nonnegativity.
- If a residual matrix is used in the matching algorithm, check that its row and column sums are equal; they need not still equal one.

## 20.3 Quick derivation checklist for Hedge

When asked to derive the regret bound, reproduce this chain:

1. Write $\Phi^t=\sum_iw_i^t$.
2. Express $\Phi^{t+1}/\Phi^t$ using $p_i^t$.
3. Apply the second-order exponential upper bound.
4. Take logarithms.
5. Sum over rounds and telescope.
6. Lower-bound $\Phi^{T+1}$ using the weight of the best fixed action.
7. Combine upper and lower bounds.
8. Use $(\ell_i^t)^2\le B^2$.
9. Optimize $\eta$.

The most common derivation error is losing the negative sign in $e^{-\eta\ell_i^t}$, or forgetting that the second-order term contains the **squared loss**.

---

# 21. Short revision summary

The four-week progression can be remembered as a sequence of increasingly general solutions.

1. **Majority:** If a perfect expert exists, follow the majority of the surviving experts and eliminate those who are wrong. The number of mistakes is logarithmic in the number of experts.

2. **Regret:** If no expert is perfect, compare the learner against the best expert in hindsight rather than demanding zero mistakes.

3. **Weighted Majority:** Maintain weights and reduce the influence of experts who make mistakes.

4. **Randomization:** Against an oblivious adversary, randomization hides the realized action even when the algorithm and its probabilities are known.

5. **Hedge:** Generalize multiplicative weights to arbitrary bounded losses and actions represented by a probability simplex. Its regret is $O(\sqrt{T\ln d})$.

6. **Combinatorial actions:** Ordinary Hedge becomes impractical when the number of actions is $n!$.

7. **Permutation matrices:** Represent assignments using $n\times n$ matrices with one 1 in each row and column.

8. **Doubly stochastic matrices:** Represent distributions over assignments using a single matrix in the Birkhoff polytope.

9. **Matching decomposition:** Decompose a doubly stochastic matrix into a small number of permutation matrices by repeatedly finding a perfect matching and subtracting its minimum edge weight.

10. **Combinatorial Hedge:** Maintain a weight matrix, balance it using Sinkhorn scaling, decompose it, sample an assignment, and update the weights exponentially.

The recurring design pattern is:

$$
\boxed{
\text{Represent actions}
\longrightarrow
\text{maintain weights}
\longrightarrow
\text{normalize}
\longrightarrow
\text{choose an action}
\longrightarrow
\text{observe losses}
\longrightarrow
\text{update weights}.
}
$$

The central lesson is that a good sequential algorithm must satisfy two requirements:

- **Statistical performance:** small regret compared with the best fixed action.
- **Computational feasibility:** the representation and updates must remain manageable even when the action space is very large.

---

# 22. Suggested self-test before the quiz

Try answering these without looking at the notes:

1. Why does the Majority Algorithm make at most logarithmically many mistakes when a perfect expert exists?
2. What is the difference between an upper bound and a lower bound on mistakes?
3. Why is regret defined against the best expert in hindsight?
4. What exactly does an oblivious adversary know and not know?
5. Why does randomization help against an oblivious adversary?
6. How does the expected loss of a sampled action become a dot product?
7. Why is the best fixed comparator for a linear loss over a simplex always a vertex?
8. Derive the Hedge regret bound from the ratio of consecutive total weights.
9. Why must the learning rate depend on $T$?
10. Why is maintaining $n!$ weights impractical for the assignment problem?
11. What is the difference between a permutation matrix and a doubly stochastic matrix?
12. Why does every doubly stochastic matrix admit a decomposition into permutation matrices?
13. Why does the matching decomposition always find a perfect matching?
14. Why does each iteration eliminate at least one positive entry?
15. Why is the matching decomposition polynomial-time?

If you can explain the reasoning behind these fifteen questions—not merely recall the formulas—you have covered the main mathematical structure of the first four weeks.

---

## Source alignment

- Lecture transcript: `L2-23.txt` — introductory decision-making settings, expert advice, Majority, regret, Weighted Majority, randomization, Hedge, assignment problem, and matrix decomposition.
- Professor's slide: `week-4-1.pdf` — Hedge algorithm, regret definition, exponential inequalities, telescoping total-weight proof, and optimized regret bound.

*These notes are a structured study guide and include a few mathematical clarifications where a precise distinction is useful. They are not a verbatim transcript.*
