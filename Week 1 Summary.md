Absolutely. I went through the Week 1 transcript as a **course revision document**, rather than merely shortening the lecture. The central story of the week is actually quite elegant:

> **Start with minimizing mistakes → discover the “perfect expert” assumption is unrealistic → replace absolute performance with regret → discover even regret can be impossible under a fully adaptive adversary → weaken the adversary using the notion of an oblivious adversary.**

That progression is probably the most important thing to remember for the exam.

# Week 1: Online Learning and Learning from Expert Advice

The lectures begin by placing **online learning** in the broader family of sequential decision-making problems. The course distinguishes online learning, multi-armed bandits, and reinforcement learning according to the amount and type of feedback available. The Week 1 focus is **online learning**, particularly **learning from expert advice**. L2-7

---

# 1. Big Picture

### Sequential decision-making

At every round:

1. We make a decision.
2. The environment reveals feedback.
3. We use that feedback to make a better decision next time.

For this course, the simplified example is:

**Experts → Algorithm → Prediction → Truth/Feedback → Update**

For example:

```text
        Expert 1 ──┐
        Expert 2 ──┤
        Expert 3 ──┤
        ...        ├──> Algorithm ──> Prediction ŷt
        Expert D ──┘                       │
                                           ↓
                                     Actual outcome yt
                                           │
                                           ↓
                                   Update algorithm
```

The experts provide binary advice:

- `1` = Invest
- `0` = Don't invest

The algorithm must combine these recommendations.

---

# 2. Online Learning vs Other Sequential Learning Settings

| Setting | Feedback received | Important characteristic |
|---|---|---|
| **Online learning** | Full information | We can see what happened to all experts' predictions |
| **Multi-armed bandit** | Partial information | We mainly observe the outcome of the action we selected |
| **Reinforcement learning** | Partial information + state transitions | Actions influence future states |

The Week 1 lectures concentrate on **online learning / full-information learning**. L2-7

### Exam memory trick

**Online → see everything**

**Bandit → see selected action**

**RL → see selected action + state changes**

---

# 3. Learning from Expert Advice

Suppose there are **D experts**.

At round \(t\):

- Expert \(1,\ldots,D\) gives advice.
- Algorithm observes all advice.
- Algorithm predicts.
- Actual outcome \(y_t\) is revealed.
- Algorithm updates its strategy.

Let

$$
x_t=(x_{t1},x_{t2},\ldots,x_{tD})
$$

where

$$
x_{tk}\in\{0,1\}
$$

is the prediction of expert \(k\) at time \(t\).

The algorithm produces

$$
\hat y_t\in\{0,1\}
$$

and the actual outcome is

$$
y_t\in\{0,1\}.
$$

The transcript explicitly represents the experts' advice as a binary vector \(x_t\in\{0,1\}^D\). L2-7

---

# 4. What Is a Mistake?

The algorithm makes a mistake whenever

$$
\boxed{\hat y_t\neq y_t}
$$

Therefore the total number of mistakes over \(T\) rounds is

$$
\boxed{
M_A(T)=
\sum_{t=1}^{T}
\mathbf 1[\hat y_t\neq y_t]
}
$$

where

$$
\mathbf 1[\text{condition}]
=
\begin{cases}
1,&\text{condition true}\\
0,&\text{otherwise}
\end{cases}
$$

### Important interpretation

| Prediction | Reality | Mistake? |
|---|---|---|
| 1 | 1 | No |
| 0 | 0 | No |
| 1 | 0 | Yes |
| 0 | 1 | Yes |

This binary mistake/loss formulation is the foundation for everything that follows.

---

# 5. First Setup: The Perfect "Guru"

Initially the lecturer assumes:

> Among the \(D\) experts, **at least one expert is always correct**.

This expert is called the **stock-market guru**.

The algorithm knows:

$$
\boxed{\text{There exists a perfect expert}}
$$

but **does not know which expert it is**. L2-7

This assumption is crucial.

---

# 6. Majority Algorithm

The natural strategy is:

> Follow the majority of the experts.

Suppose:

- 70 experts say Invest
- 30 experts say Don't Invest

Then

$$
\hat y_t=1
$$

because the majority says 1.

After the actual result is revealed, eliminate experts who were wrong.

### Algorithm

Initially:

$$
C_1=\{1,2,\ldots,D\}
$$

At each round:

1. Consider the current candidate set \(C_t\).
2. Take the majority prediction.
3. Make that prediction.
4. Observe \(y_t\).
5. Remove all experts who made a mistake.

So:

$$
C_{t+1}
=
\{k\in C_t:x_{tk}=y_t\}
$$

The lecture calls this the **majority algorithm**. L2-7

---

# 7. Why Does the Majority Algorithm Work?

Suppose the algorithm makes a mistake.

Since the algorithm followed the majority, the **majority must have been wrong**.

Therefore at least half of the current experts are eliminated.

So:

$$
|C_{t+1}|
\leq
\frac{|C_t|}{2}
$$

After \(m\) mistakes:

$$
|C|
\leq
\frac{D}{2^m}
$$

Eventually only one expert remains.

We need

$$
\frac{D}{2^m}<1
$$

which gives approximately

$$
2^m>D
$$

and hence

$$
m=O(\log D)
$$

More precisely, the lecture gives the worst-case mistake bound as

$$
\boxed{M_A\leq\lceil\log_2D\rceil}
$$

under the perfect-expert assumption. L2-7

---

# 8. Why the Bound Is Logarithmic

This is a very important exam intuition.

Every mistake cuts the candidate population approximately in half:

$$
D
\rightarrow
\frac D2
\rightarrow
\frac D4
\rightarrow
\frac D8
\rightarrow\cdots
$$

After \(m\) mistakes:

$$
\frac{D}{2^m}
$$

Set this equal to 1:

$$
\frac{D}{2^m}=1
$$

Therefore:

$$
2^m=D
$$

so

$$
\boxed{m=\log_2D}
$$

### Example

If

$$
D=1,000,000
$$

then

$$
\log_2(1,000,000)\approx19.93
$$

so at most about **20 mistakes** are required in the worst case.

That is the beautiful part of the argument:

> **A million experts do not mean a million mistakes. The candidate set shrinks exponentially.**

---

# 9. Is the Majority Algorithm Optimal?

Now the lecture asks a deeper question:

> Could another algorithm make fewer than \(O(\log D)\) mistakes?

To answer this, we need a **lower bound**.

An upper bound says:

$$
\text{This algorithm makes at most }f(D)\text{ mistakes.}
$$

A lower bound says:

$$
\text{Every possible algorithm can be forced to make at least }f(D)\text{ mistakes.}
$$

The latter is much stronger.

---

# 10. Adversarial Lower Bound

Imagine an adversary controlling:

- the experts' advice
- the actual outcomes

but with one restriction:

> At least one expert must remain perfectly correct.

The adversary can arrange approximately:

$$
D/2 \text{ experts say 0}
$$

and

$$
D/2 \text{ experts say 1}
$$

Whatever the algorithm predicts, the adversary chooses the opposite truth.

That causes a mistake.

But now the perfect expert must be among the experts who predicted correctly.

Therefore the candidate set shrinks by at most half.

The adversary can repeat this approximately

$$
\log_2D
$$

times.

Therefore:

$$
\boxed{
\text{Any algorithm can be forced to make }
\Omega(\log D)\text{ mistakes}
}
$$

The majority algorithm already achieves

$$
O(\log D)
$$

so we obtain:

$$
\boxed{
\Theta(\log D)
}
$$

This establishes optimality in this particular perfect-expert setting. L2-7

---

# 11. Why the Perfect Expert Assumption Is Unrealistic

The lecturer then identifies the elephant in the room:

> In reality, there is generally no perfect expert.

An expert can make mistakes.

Therefore, if an expert makes one mistake, we **cannot permanently discard that expert**.

Why?

Because that expert could perform extremely well over the next 100 rounds.

This destroys the logic of the previous majority algorithm. L2-7

---

# 12. The New Goal: Regret

This is probably the **single most important concept of Week 1**.

If no expert is perfect, minimizing our absolute number of mistakes is impossible.

So instead we ask:

> **How much worse are we than the best expert?**

This difference is called **regret**.

---

# 13. Mistakes of the Algorithm

Define:

$$
M_A(T)
=
\sum_{t=1}^{T}
\mathbf1[\hat y_t\neq y_t]
$$

where \(M_A(T)\) is the number of mistakes made by the algorithm.

For expert \(k\):

$$
M_k(T)
=
\sum_{t=1}^{T}
\mathbf1[x_{tk}\neq y_t]
$$

The best expert in hindsight is:

$$
M^*(T)
=
\min_{k\in\{1,\ldots,D\}}M_k(T)
$$

Therefore:

$$
\boxed{
R_A(T)
=
M_A(T)-M^*(T)
}
$$

This is **regret**. L2-7

---

# 14. Why Is It Called "Regret"?

Suppose after 10,000 rounds:

- Algorithm: 4,000 mistakes
- Best expert: 400 mistakes

Then:

$$
R_A(10000)
=
4000-400
=
3600
$$

Interpretation:

> "If only I had followed that expert from the beginning!"

That is why the quantity is called **regret**. L2-7

---

# 15. Best Expert Is Chosen in Hindsight

This is a subtle but **very exam-worthy** point.

The best expert is:

$$
k^*
=
\arg\min_k M_k(T)
$$

after observing all \(T\) rounds.

So the identity of the best expert can change when \(T\) changes.

For example:

| Time | Best expert so far |
|---|---|
| \(T=100\) | Expert 52 |
| \(T=10,000\) | Expert 48 |
| \(T=20,000\) | Expert 17 |

The algorithm is always compared against:

$$
\boxed{\text{best expert in hindsight}}
$$

not necessarily one fixed expert known beforehand. L2-7

---

# 16. What Is a Good Algorithm?

We don't necessarily demand:

$$
R_A(T)\rightarrow0
$$

because total regret can naturally increase as more rounds are played.

Instead we ask for **average regret** to vanish:

$$
\boxed{
\frac{R_A(T)}{T}\rightarrow0
\quad\text{as }T\rightarrow\infty
}
$$

This is called **sublinear regret**.

Equivalently:

$$
R_A(T)=o(T)
$$

This means the regret grows slower than linearly.

The lecture explicitly identifies vanishing per-round regret and sublinear regret as the desired criterion. L2-7

---

# 17. Linear vs Sublinear Regret

| Regret | Average regret | Good? |
|---|---:|---|
| \(R(T)=T\) | \(1\) | ❌ |
| \(R(T)=T/2\) | \(1/2\) | ❌ |
| \(R(T)=\sqrt T\) | \(1/\sqrt T\to0\) | ✅ |
| \(R(T)=\log T\) | \(\log T/T\to0\) | ✅ |
| \(R(T)=100\) | \(100/T\to0\) | ✅ |

### Key equivalence

$$
\boxed{
R(T)=o(T)
\iff
\frac{R(T)}T\rightarrow0
}
$$

---

# 18. Sanity Check: Return to the Guru Case

Suppose a perfect expert exists.

Then:

$$
M^*(T)=0
$$

Therefore:

$$
R_A(T)=M_A(T)
$$

For the majority algorithm:

$$
R_A(T)\leq\log_2D
$$

Thus:

$$
\frac{R_A(T)}T
\leq
\frac{\log_2D}{T}
\rightarrow0
$$

So the old majority algorithm is indeed a **zero-average-regret algorithm** in the special guru setting. L2-7

This is a useful conceptual bridge:

> **Regret generalizes the old mistake-minimization objective.**

---

# 19. Why We Need a New Algorithm

We cannot simply eliminate an expert after one mistake.

Instead:

> Keep all experts, but give better experts more influence.

This leads naturally to **weights**.

Each expert gets a weight:

$$
w_{t,k}
$$

Initially:

$$
\boxed{w_{1,k}=1}
$$

for every expert \(k\).

So initially:

$$
W_1=\sum_{k=1}^{D}w_{1,k}=D
$$

The lecture introduces this weighted approach precisely because experts cannot safely be eliminated. L2-7

---

# 20. Weighted Majority Algorithm

At round \(t\):

### Step 1: Receive advice

$$
x_t=(x_{t1},...,x_{tD})
$$

### Step 2: Calculate weighted support

Weight supporting prediction 1:

$$
S_1=
\sum_{k:x_{tk}=1}w_{t,k}
$$

Weight supporting prediction 0:

$$
S_0=
\sum_{k:x_{tk}=0}w_{t,k}
$$

### Step 3: Predict weighted majority

$$
\boxed{
\hat y_t=
\begin{cases}
1,&S_1>S_0\\
0,&S_0>S_1
\end{cases}
}
$$

Ties can be broken arbitrarily. L2-7

---

# 21. Weight Update

Let

$$
0<\epsilon<1
$$

be the weight reduction parameter.

If expert \(k\) makes a mistake:

$$
x_{tk}\neq y_t
$$

then:

$$
\boxed{
w_{t+1,k}
=
(1-\epsilon)w_{t,k}
}
$$

If expert \(k\) is correct:

$$
x_{tk}=y_t
$$

then:

$$
\boxed{
w_{t+1,k}=w_{t,k}
}
$$

So:

> **Good prediction → weight unchanged**

> **Bad prediction → weight reduced**

The lecture gives exactly this update rule and calls the method the **Weighted Majority Algorithm**. L2-7

---

# 22. Meaning of \(\epsilon\)

$$
\epsilon
$$

controls how aggressively we punish a wrong expert.

Example:

$$
\epsilon=0.1
$$

Then:

$$
w_{t+1,k}=0.9w_{t,k}
$$

So the expert loses **10% of its weight** whenever it makes a mistake.

### Extreme cases

$$
\epsilon=0
$$

means:

> Nobody is ever punished.

Not useful.

Whereas

$$
\epsilon=1
$$

means:

$$
w_{t+1,k}=0
$$

for a mistaken expert.

That reduces Weighted Majority to the earlier **elimination/majority algorithm**. L2-7

This connection is very important:

$$
\boxed{\epsilon=1\Rightarrow\text{Weighted Majority becomes Majority/Elimination}}
$$

---

# 23. Potential Function Analysis

The lecturer then analyzes Weighted Majority using a **potential function**.

Define:

$$
\boxed{
\Phi_t=\sum_{k=1}^{D}w_{t,k}
}
$$

This is simply the **total weight of all experts**.

Initially:

$$
\boxed{\Phi_1=D}
$$

because every expert begins with weight 1. L2-7

---

# 24. Key Potential-Function Observation

Suppose the algorithm makes a mistake.

The algorithm followed the **weighted majority**.

Therefore the experts who were wrong had at least half of the total weight:

$$
\sum_{k:x_{tk}\neq y_t}w_{t,k}
\geq
\frac{\Phi_t}{2}
$$

Those experts are penalized by \(1-\epsilon\).

Therefore:

$$
\Phi_{t+1}
\leq
\Phi_t
\left(1-\frac{\epsilon}{2}\right)
$$

This is one of the most important derivations in the lecture. L2-7

---

# 25. After \(M\) Algorithm Mistakes

If the algorithm makes \(M\) mistakes, repeatedly applying the previous result gives:

$$
\boxed{
\Phi_T
\leq
D
\left(1-\frac{\epsilon}{2}\right)^M
}
$$

This connects:

- number of experts \(D\)
- learning/penalty parameter \(\epsilon\)
- algorithm mistakes \(M\)

The lecture derives this potential upper bound explicitly. L2-7

---

# 26. Lower Bound on Potential Using the Best Expert

Let the best expert make

$$
M^*
$$

mistakes.

That expert starts with weight 1.

Every time it makes a mistake, its weight gets multiplied by \(1-\epsilon\).

Therefore:

$$
\boxed{
w_T^*
=
(1-\epsilon)^{M^*}
}
$$

Since total weight contains the best expert:

$$
\Phi_T\geq w_T^*
$$

so:

$$
\boxed{
\Phi_T\geq(1-\epsilon)^{M^*}
}
$$

The lecturer calls this the second side of the "sandwich". L2-7

---

# 27. The Famous Sandwich

Putting the two bounds together:

$$
\boxed{
(1-\epsilon)^{M^*}
\leq
\Phi_T
\leq
D\left(1-\frac{\epsilon}{2}\right)^M
}
$$

This is the heart of the Weighted Majority analysis.

It gives us a bridge between:

- \(M\): algorithm mistakes
- \(M^*\): best expert's mistakes
- \(D\): number of experts
- \(\epsilon\): penalty parameter

---

# 28. Useful Mathematical Inequalities

The lecturer introduces:

$$
\boxed{e^x\geq1+x}
$$

Therefore:

$$
e^{-x}\geq1-x
$$

and taking logarithms:

$$
\boxed{\log(1-x)\leq-x}
$$

Another inequality used is:

$$
\boxed{
\log(1-x)\geq-x-x^2
}
$$

for the range stated in the lecture, \(0\leq x\leq\frac12\). L2-7

### Exam tip

Remember the direction:

$$
\boxed{\log(1-x)\leq -x}
$$

and, for small \(x\),

$$
\boxed{\log(1-x)\approx-x-\frac{x^2}{2}}
$$

---

# 29. Weighted Majority Regret Bound Obtained in the Lecture

After the algebra, the lecture obtains:

$$
\boxed{
M-M^*
\leq
M^*
+
\frac{2\log D}{\epsilon}
+
2\epsilon M^*
}
$$

or, in regret notation,

$$
\boxed{
R_{WM}(T)
\leq
M^*
+
\frac{2\log D}{\epsilon}
+
2\epsilon M^*
}
$$

The transcript explicitly identifies \(M-M^*\) as regret and gives this resulting bound. L2-7

### Important interpretation

The bound contains a term involving

$$
M^*
$$

which can itself grow linearly with \(T\).

Therefore this particular bound **does not establish zero-average/sublinear regret**.

That does **not automatically mean the algorithm is bad**. It means the bound obtained so far does not give the desired guarantee. L2-7

---

# 30. The Three Possible Explanations

The lecture asks why the desired sublinear guarantee was not obtained.

There are three possibilities:

| Possibility | Meaning |
|---|---|
| 1 | Weighted Majority is intrinsically bad |
| 2 | The analysis/bound is not tight enough |
| 3 | The problem itself makes sublinear regret impossible |

The lecture resolves this through a two-expert adversarial example. L2-7

---

# 31. Two-Expert Impossibility Example

Take only two experts:

### Expert 1

Always predicts:

$$
1
$$

### Expert 2

Always predicts:

$$
0
$$

Now imagine an adversary that observes the algorithm's prediction and always gives the opposite truth.

If the algorithm predicts:

$$
1
$$

the truth becomes:

$$
0
$$

If the algorithm predicts:

$$
0
$$

the truth becomes:

$$
1
$$

Therefore:

$$
\boxed{M_A(T)=T}
$$

The algorithm makes a mistake **every round**. L2-7

---

# 32. Why the Best Expert Is Still Not Too Bad

Let expert 1 make \(L\) mistakes.

Since expert 1 always predicts 1, expert 2 always predicts 0, and the truth is binary:

$$
M_1=L
$$

and

$$
M_2=T-L
$$

Therefore:

$$
M_1+M_2=T
$$

Hence:

$$
\min(M_1,M_2)\leq\frac T2
$$

Therefore:

$$
M^*\leq\frac T2
$$

But the algorithm has

$$
M_A=T
$$

so:

$$
R_A(T)
=
T-M^*
\geq
\frac T2
$$

Thus:

$$
\boxed{
\frac{R_A(T)}T\geq\frac12
}
$$

and therefore:

$$
\boxed{
\frac{R_A(T)}T\nrightarrow0
}
$$

So **no algorithm** can guarantee sublinear regret against this fully adaptive adversary. L2-7

This is an extremely important exam result.

---

# 33. What Actually Went Wrong?

The problem is not necessarily:

> "Weighted Majority is bad."

The deeper issue is:

> **The adversary has been given too much power.**

The adversary observes the algorithm's current decision and immediately chooses the opposite outcome.

In other words:

```text
Algorithm predicts
       ↓
Adversary observes prediction
       ↓
Adversary chooses opposite outcome
       ↓
Algorithm loses
```

This is an extraordinarily powerful adversary.

The lecturer therefore proposes **not necessarily changing the goal of regret**, but instead restricting the adversary. L2-7

---

# 34. Oblivious Adversary

This is where the lecture ends and sets up the next part of the course.

An **oblivious adversary** must choose the entire sequence of outcomes **before the game begins**.

So:

$$
y_1,y_2,\ldots,y_T
$$

is fixed in advance.

The adversary may know the algorithm, but it cannot decide \(y_t\) after seeing the algorithm's current prediction. L2-7

### Adaptive adversary

$$
\hat y_t
\rightarrow
\boxed{\text{adversary sees it}}
\rightarrow
y_t
$$

### Oblivious adversary

Before game:

$$
\boxed{
(y_1,y_2,\ldots,y_T)
\text{ fixed}
}
$$

Then game runs:

$$
y_1,y_2,\ldots,y_T
$$

This distinction is likely to become very important in the following lectures. The transcript explicitly ends by asking how this restriction can be exploited to obtain better algorithms and potentially sublinear regret. L2-7

---

# 35. The Entire Week in One Chain

This is the **most useful revision chain** to memorize:

$$
\boxed{
\text{Online Learning}
}
$$

↓

$$
\boxed{
\text{Learning from Expert Advice}
}
$$

↓

$$
\boxed{
\text{Perfect Expert Assumption}
}
$$

↓

$$
\boxed{
\text{Majority Algorithm}
}
$$

↓

$$
\boxed{
O(\log D)\text{ mistakes}
}
$$

↓

$$
\boxed{
\text{Optimal via lower bound}
}
$$

↓

$$
\boxed{
\text{But perfect expert is unrealistic}
}
$$

↓

$$
\boxed{
\text{Cannot eliminate experts}
}
$$

↓

$$
\boxed{
\text{Weighted Majority}
}
$$

↓

$$
\boxed{
\text{Compare against best expert}
}
$$

↓

$$
\boxed{
\text{Regret}
}
$$

↓

$$
\boxed{
R(T)/T\rightarrow0
}
$$

↓

$$
\boxed{
\text{Fully adaptive adversary makes this impossible}
}
$$

↓

$$
\boxed{
\text{Restrict adversary}
}
$$

↓

$$
\boxed{
\text{Oblivious Adversary}
}
$$

That is the conceptual backbone of Week 1.

---

# Key Takeaways Table

| # | Concept | Key takeaway |
|---|---|---|
| 1 | Online learning | Sequential decision-making with feedback |
| 2 | Expert advice | Multiple experts provide predictions each round |
| 3 | Full information | We see the actual outcome and can evaluate all experts |
| 4 | Mistake | \(\hat y_t\neq y_t\) |
| 5 | Perfect expert | At least one expert is always correct |
| 6 | Majority algorithm | Follow majority of currently consistent experts |
| 7 | Mistake reduction | Every mistake eliminates at least half the candidates |
| 8 | Majority bound | \(\lceil\log_2D\rceil\) mistakes |
| 9 | Lower bound | An adversary can force \(\Omega(\log D)\) mistakes |
| 10 | Optimality | Majority achieves the lower bound in the guru setting |
| 11 | Problem | Real experts are not perfect |
| 12 | Regret | Compare algorithm with best expert in hindsight |
| 13 | Best expert | \(M^*(T)=\min_kM_k(T)\) |
| 14 | Regret | \(R(T)=M_A(T)-M^*(T)\) |
| 15 | Good algorithm | \(R(T)/T\to0\) |
| 16 | Sublinear regret | \(R(T)=o(T)\) |
| 17 | Weighted Majority | Keep every expert but assign weights |
| 18 | Weight penalty | Wrong experts get multiplied by \(1-\epsilon\) |
| 19 | Potential | \(\Phi_t=\sum_k w_{t,k}\) |
| 20 | Potential reduction | Mistake implies \(\Phi_{t+1}\leq(1-\epsilon/2)\Phi_t\) |
| 21 | Problem with bound | Obtained bound still contains \(M^*\) |
| 22 | Impossibility | Fully adaptive adversary can force linear regret |
| 23 | Key cause | Adversary sees the algorithm's current prediction |
| 24 | Oblivious adversary | Outcome sequence fixed before the game |

---

# Critical Formula Sheet

| Quantity | Formula | Meaning |
|---|---|---|
| Expert prediction | \(x_{tk}\in\{0,1\}\) | Expert \(k\)'s prediction at round \(t\) |
| Algorithm prediction | \(\hat y_t\in\{0,1\}\) | Algorithm's decision |
| Actual outcome | \(y_t\in\{0,1\}\) | Truth |
| Algorithm mistakes | \(\displaystyle M_A(T)=\sum_{t=1}^T\mathbf1[\hat y_t\neq y_t]\) | Total algorithm mistakes |
| Expert \(k\)'s mistakes | \(\displaystyle M_k(T)=\sum_{t=1}^T\mathbf1[x_{tk}\neq y_t]\) | Mistakes of expert \(k\) |
| Best expert | \(\displaystyle M^*(T)=\min_kM_k(T)\) | Best expert in hindsight |
| Regret | \(\boxed{R_A(T)=M_A(T)-M^*(T)}\) | Extra mistakes over best expert |
| Zero-average regret | \(\displaystyle R_A(T)/T\to0\) | Desired asymptotic property |
| Sublinear regret | \(\boxed{R_A(T)=o(T)}\) | Equivalent condition |
| Majority candidate set | \(C_{t+1}=\{k\in C_t:x_{tk}=y_t\}\) | Retain correct experts |
| Majority reduction | \(|C_{t+1}|\leq|C_t|/2\) | After an algorithm mistake |
| Majority mistake bound | \(\boxed{M_A\leq\lceil\log_2D\rceil}\) | Perfect-expert setting |
| Weighted initialisation | \(w_{1,k}=1\) | Equal initial trust |
| Total weight | \(\displaystyle\Phi_t=\sum_kw_{t,k}\) | Potential |
| Weighted prediction | Compare \(\sum_{x_{tk}=1}w_{tk}\) and \(\sum_{x_{tk}=0}w_{tk}\) | Weighted majority |
| Weight update | \(w_{t+1,k}=(1-\epsilon)w_{t,k}\) | If expert is wrong |
| Correct expert update | \(w_{t+1,k}=w_{t,k}\) | If expert is correct |
| Potential after mistake | \(\displaystyle\Phi_{t+1}\leq(1-\epsilon/2)\Phi_t\) | Key analysis step |
| After \(M\) mistakes | \(\displaystyle\Phi_T\leq D(1-\epsilon/2)^M\) | Potential upper bound |
| Best expert weight | \(\displaystyle w_T^*=(1-\epsilon)^{M^*}\) | Lower bound ingredient |
| Potential lower bound | \(\displaystyle\Phi_T\geq(1-\epsilon)^{M^*}\) | Best expert is included |
| Sandwich | \(\displaystyle(1-\epsilon)^{M^*}\leq\Phi_T\leq D(1-\epsilon/2)^M\) | Core WM analysis |
| Lecture's regret bound | \(\displaystyle R_{WM}\leq M^*+\frac{2\log D}{\epsilon}+2\epsilon M^*\) | Bound obtained in lecture |
| Two-expert impossibility | \(M_A=T,\ M^*\leq T/2\) | Fully adaptive adversary |
| Result | \(\displaystyle R_A(T)\geq T/2\) | No sublinear regret in that setting |

The potential-function and regret calculations above follow the lecture's derivation. L2-7 L2-7

---

# Critical Distinctions for the Exam

These are the places where an MCQ can quietly set a trap.

### 1. Mistakes vs regret

**Mistakes:**

$$
M_A(T)
$$

asks:

> How many times did my algorithm get it wrong?

**Regret:**

$$
R_A(T)=M_A(T)-M^*(T)
$$

asks:

> How much worse was I than the best expert in hindsight?

---

### 2. Upper bound vs lower bound

**Upper bound:**

$$
M_A(T)\leq f(T)
$$

says a particular algorithm cannot do worse than \(f(T)\).

**Lower bound:**

$$
M_A(T)\geq f(T)
$$

under an adversarial construction says an algorithm cannot universally do better than \(f(T)\).

For the guru setting:

$$
\boxed{\text{Upper bound}=\Theta(\log D)}
$$

and

$$
\boxed{\text{Lower bound}=\Theta(\log D)}
$$

therefore the majority algorithm is optimal in that setting.

---

### 3. Perfect expert vs best expert

These are **not the same assumption**.

**Perfect expert:**

$$
M^*=0
$$

**Best expert:**

$$
M^* \geq0
$$

The best expert may make many mistakes.

This distinction is what motivates regret.

---

### 4. Majority vs Weighted Majority

| Majority | Weighted Majority |
|---|---|
| Experts can be eliminated | Experts are retained |
| Wrong expert can get weight 0 | Wrong expert gets reduced weight |
| Suitable for perfect-expert assumption | Designed for imperfect experts |
| Hard elimination | Soft penalty |
| Corresponds to \(\epsilon=1\) | Usually \(0<\epsilon<1\) |

---

### 5. Adaptive vs oblivious adversary

**Adaptive:**

$$
\boxed{\text{Adversary can react to current algorithm prediction}}
$$

**Oblivious:**

$$
\boxed{\text{Outcome sequence fixed before the game}}
$$

The fully adaptive adversary is powerful enough to force:

$$
R(T)\geq T/2
$$

in the two-expert construction.

---

# What I Would Memorize for the Exam

If you have only five minutes before entering the exam hall, remember these **10 statements**:

1. **Online learning = sequential decisions + feedback.**
2. **Learning from expert advice = combine predictions of \(D\) experts.**
3. **Perfect expert assumption → Majority Algorithm.**
4. **Every mistake eliminates at least half the candidate experts.**
5. Therefore:
   $$
   \boxed{M\leq\lceil\log_2D\rceil}
   $$
6. Without a perfect expert, don't eliminate experts. **Weight them.**
7. **Regret = algorithm mistakes − best expert's mistakes.**
   $$
   \boxed{R=M-M^*}
   $$
8. Desired:
   $$
   \boxed{R(T)/T\to0}
   $$
   or
   $$
   \boxed{R(T)=o(T)}
   $$
9. **Weighted Majority:** wrong expert's weight gets multiplied by \(1-\epsilon\).
10. A fully adaptive adversary can force linear regret, motivating the **oblivious adversary** model.

---

# One-Page Mental Map

```text
                 ONLINE LEARNING
                       │
                       ↓
           LEARNING FROM EXPERT ADVICE
                       │
                       ↓
             D experts give advice
                       │
             ┌─────────┴─────────┐
             ↓                   ↓
       Perfect expert?       No perfect expert
             │                   │
             ↓                   ↓
       Majority Algorithm    Can't eliminate experts
             │                   │
             ↓                   ↓
       Mistake ≤ log₂D       Weighted Majority
             │                   │
             ↓                   ↓
       Is it optimal?        Compare with best expert
             │                   │
             ↓                   ↓
       Lower bound            REGRET
       Ω(log D)                 │
             │                  ↓
             └────────────→ R(T)=M-M*
                                │
                                ↓
                         Want R(T)/T → 0
                                │
                                ↓
                   Fully adaptive adversary
                                │
                                ↓
                   Can force R(T) ≥ T/2
                                │
                                ↓
                    Sublinear regret impossible
                                │
                                ↓
                     Restrict adversary
                                │
                                ↓
                    OBLIVIOUS ADVERSARY
                                │
                                ↓
                  Next part of the course
```

## One important note about the "extra context" formulas

I have kept the mathematical results **faithful to the transcript**, including the lecture's specific Weighted Majority bound. I have not silently replaced it with a later, more refined Hedge/exponential-weights result, because that could blur an important distinction in your course: **the lecture deliberately reaches a negative result first and then changes the adversary model.** The transcript itself ends at that transition. L2-7

So for exam preparation, I would treat **oblivious adversary** as the bridge into the next set of lectures, rather than mixing Week 1 with later algorithms.

Would you like the next revision layer as **(A) exam-style MCQs, (B) numerical problems, or (C) a 2-page formula cheat sheet**?
