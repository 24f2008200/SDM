# Week 2 Lecture Notes: Online Learning and Learning from Expert Advice

The most useful way to study this week's material is to follow the evolution of the central question:

How can an algorithm learn from expert advice when the environment is adversarial, and how can we guarantee that its performance improves over time?

Week 1 introduced regret, Weighted Majority, and the difficulty caused by an adaptive adversary. Week 2 builds on that foundation, exploring how randomization, expected regret, and carefully chosen learning parameters help us obtain stronger guarantees.

I'll organize the notes into concepts, derivations, key takeaways, and a formula sheet, with a few additional mathematical connections useful for exam revision.

## 1. The learning problem and its performance measure

We have $D$ experts. At every round $t$:

- Expert $k$ gives advice $x\_{tk}\in\\{0,1\\}$.
- The algorithm predicts $\hat y_t$.
- The actual outcome $y_t$ is revealed.
- The algorithm updates its expert weights.

The key difference from the perfect-guru setting is that no expert is assumed to be perfect.

### Core definitions

The algorithm's number of mistakes is

$$ M_A(T)=\sum\_{t=1}^{T} \mathbf 1[\hat y_t\ne y_t]. $$

The mistakes made by expert $k$ are

$$ M_k(T)=\sum\_{t=1}^{T} \mathbf 1[x\_{tk}\ne y_t]. $$

The best expert in hindsight makes

$$ M^\*(T)=\min\_{1\le k\le D}M_k(T). $$

Regret is the difference:

$$ \boxed{R_A(T)=M_A(T)-M^\*(T)} $$

The objective is not necessarily to make few mistakes in absolute terms. It is to avoid performing substantially worse than the best expert.

A standard target is sublinear regret:

$$ \boxed{\frac{R_A(T)}{T}\longrightarrow0} $$

This means that the average additional mistakes per round approach zero.

## 2. Why the adversary model matters

Week 1 showed that a fully adaptive adversary can observe the algorithm's prediction and then choose the opposite outcome. With two experts that always predict opposite binary labels, this can force the algorithm to make a mistake every round.

Randomization changes the situation, but there is a subtle distinction between two adversaries.

| Adversary      | When are outcomes chosen?                          | Can it react to the current random prediction? |
| -------------- | -------------------------------------------------- | ---------------------------------------------- |
| Fully adaptive | After seeing the current prediction                | Yes                                            |
| Oblivious      | The complete outcome sequence is fixed before play | No                                             |

An oblivious adversary can know the algorithm's description. What it cannot do is choose the current outcome after observing the algorithm's random realization.

Exam trap: Knowing the algorithm is not the same as knowing its next random prediction.

## 3. From Weighted Majority to Randomized Weighted Majority

The deterministic Weighted Majority algorithm assigns each expert a weight $w\_{t,k}$. The weight represents the influence of that expert on the algorithm's decision.

Initially,

$$ w\_{1,k}=1. $$

When an expert makes a mistake, its weight is reduced:

$$ w\_{t+1,k}= \begin{cases} w\_{t,k},&x\_{tk}=y_t,\\\ (1-\epsilon)w\_{t,k},&x\_{tk}\ne y_t. \end{cases} $$

Here, $0<\epsilon<1$ is the penalty parameter.

The deterministic version predicts the weighted majority. The randomized version instead samples a prediction according to the weights supporting each label.

### The randomized prediction rule

Let

$$ W_t=\sum\_{k=1}^{D}w\_{t,k} $$

be the total weight.

The probability of predicting 1 is

$$ \boxed{ p_t=\Pr(\hat y_t=1) =\frac{\sum\_{k:x\_{tk}=1}w\_{t,k}} {\sum\_{k=1}^{D}w\_{t,k}} } $$

Similarly,

$$ \boxed{\Pr(\hat y_t=0)=1-p_t} $$

Thus, if experts with total weight 8 predict 1 and experts with total weight 2 predict 0, the algorithm predicts 1 with probability $0.8$ and 0 with probability $0.2$.

This is the central algorithmic change in Week 2: weights determine probabilities, not just a majority vote.

### Worked example

Suppose three experts have weights $2,\ 1,\ 0.25$, and their advice is $1,\ 0,\ 1$, respectively.

The total weight is

$$ W_t=2+1+0.25=3.25. $$

The experts predicting 1 have total weight $2+0.25=2.25$.

Therefore,

$$ p_t=\frac{2.25}{3.25} =\frac9{13}\approx0.6923. $$

So the algorithm predicts 1 with probability approximately $69.23\\%$.

Notice that the algorithm does not always follow the highest-weight side. It preserves some probability of choosing the other side, which is precisely what makes the prediction randomized.

## 4. Random regret and expected regret

Once predictions are randomized, regret itself becomes a random variable.

The algorithm might make different predictions on different runs, even when the expert advice and outcomes are identical.

Therefore, we use expected regret:

$$ \boxed{ \mathbb E[R_A(T)] = \mathbb E[M_A(T)]-M^\*(T) } $$

This expression assumes that the expert advice and outcome sequence are fixed independently of the algorithm's random choices, as in the oblivious-adversary model.

Why does the expectation simplify this way?

Because $M^\*(T)$ is fixed for a fixed sequence of outcomes and expert advice. It does not depend on the algorithm's random prediction.

For a randomized prediction with probability $p_t$ of predicting 1, the expected mistake indicator is

$$ \mathbb E[\mathbf1[\hat y_t\ne y_t]] = \begin{cases} 1-p_t,&y_t=1,\\\ p_t,&y_t=0. \end{cases} $$

Equivalently, using binary labels,

$$ \boxed{ \mathbb E[\text{mistake at }t] =p_t(1-y_t)+(1-p_t)y_t } $$

This formula is useful for numerical questions about randomized predictions.

## 5. Expected regret of Randomized Weighted Majority

The key result developed in the lecture is a bound of the form

$$ \boxed{ \mathbb E[R\_{\mathrm{RWM}}(T)] \leq \frac{\ln D}{\epsilon} +\epsilon M^\*(T) } $$

under the lecture's loss convention and suitable conditions on $\epsilon$.

The two terms express a trade-off:

- $\frac{\ln D}{\epsilon}$: the cost associated with distinguishing among $D$ experts.
- $\epsilon M^\*(T)$: the cost associated with penalizing experts that sometimes make mistakes.

A very small $\epsilon$ makes the first term large. A very large $\epsilon$ makes the second term large.

The advantage of randomization is that this bound compares expected mistakes directly against the best expert's mistakes, without the extra $M^\*(T)$ term that appeared in the earlier deterministic analysis.

### Choosing the penalty parameter

Suppose the horizon $T$ is known in advance. Since $M^\*(T)\leq T$, we obtain

$$ \mathbb E[R(T)] \leq \frac{\ln D}{\epsilon}+\epsilon T. $$

Choose

$$ \boxed{\epsilon=\sqrt{\frac{\ln D}{T}}} $$

to balance the two terms. Substituting:

$$ \frac{\ln D}{\epsilon} =\sqrt{T\ln D}, $$

and

$$ \epsilon T=\sqrt{T\ln D}. $$

Therefore,

$$ \boxed{ \mathbb E[R(T)] \leq 2\sqrt{T\ln D} } $$

for the parameter range where the stated bound and choice of $\epsilon$ apply.

This is a major result: the expected regret grows as $\sqrt T$, rather than linearly in $T$.

Dividing by $T$,

$$ \boxed{ \frac{\mathbb E[R(T)]}{T} \leq 2\sqrt{\frac{\ln D}{T}} \longrightarrow0 } $$

for fixed $D$.

That is the desired sublinear expected regret guarantee.

## 6. Why the $\sqrt{T}$ regret rate is significant

The lecture also considers a lower-bound argument using two deliberately simple experts:

- Expert 1 always predicts 1.
- Expert 2 always predicts 0.

Imagine the outcomes are generated by independent fair coin tosses.

The algorithm cannot predict the coin toss better than chance on average, so its expected number of mistakes is $T/2$. However, the better of the two experts benefits from random fluctuations in the sequence. Its expected number of mistakes is roughly $T/2-c\sqrt T$, for a positive constant $c$.

Consequently, the expected regret is at least of order $\sqrt T$.

The important asymptotic conclusion is

$$ \boxed{\mathbb E[R(T)]=\Omega(\sqrt T)} $$

for the worst-case problem considered in the lecture.

Compare this with the randomized algorithm's upper bound:

$$ \boxed{\mathbb E[R(T)]=O(\sqrt{T\ln D})} $$

For a fixed number of experts $D$, both bounds have square-root dependence on $T$. Thus, Randomized Weighted Majority achieves the optimal order of growth in the number of rounds, up to constants and the dependence on $D$.

This is a beautiful result: randomization does not eliminate regret, but it can bring regret down to the best achievable asymptotic scale.

## 7. The unknown-horizon problem

There is a practical difficulty with the optimal parameter choice:

$$ \epsilon=\sqrt{\frac{\ln D}{T}}. $$

What if the algorithm does not know how many rounds it will run?

For example, it may be impossible to know in advance whether a market-prediction task will run for 100 days or 100,000 days.

We cannot directly set $\epsilon$ using an unknown final horizon.

The lecture addresses this with a technique called the doubling trick.

### The doubling trick

Divide time into blocks whose lengths double:

Block 1

### 1 round

Run the algorithm, then reset.

Block 2

### 2 rounds

Reset and start a fresh run.

Block 3

### 4 rounds

Reset again after the block.

Block 4

### 8 rounds

Continue with progressively longer blocks.

The pattern continues with block lengths 16, 32, 64, and so on.

For a block of length $L$, choose

$$ \epsilon_L=\sqrt{\frac{\ln D}{L}}. $$

The regret bound for that block is then of order

$$ 2\sqrt{L\ln D}. $$

If the total number of rounds is $T$, the blocks completed by time $T$ have geometrically increasing lengths. Summing their regret bounds gives a total of order

$$ \boxed{O(\sqrt{T\ln D})}. $$

The doubling trick therefore preserves the desired sublinear rate without requiring advance knowledge of $T$.

### Why does the trick work?

The block lengths grow geometrically, so the total regret is dominated by the final few blocks rather than by a large number of equally costly blocks.

This is a general algorithm-design technique, not merely a trick for expert advice.

Remember: reset the algorithm at each block boundary, and tune the learning parameter for that block's known length.

## 8. Generalization: from expert advice to broader learning problems

The final lectures begin moving beyond the simple binary prediction example toward more general online-learning problems.

The reusable structure is:

1. Maintain a weight or score for each available choice.
2. Use those weights to select an action.
3. Observe feedback.
4. Update the weights.
5. Compare cumulative performance against a benchmark.

The binary expert setting uses mistakes as its loss function. A broader formulation assigns a loss $\ell\_{t,k}$ to each expert or action.

For example, in a general loss-based setting:

$$ L_k(T)=\sum\_{t=1}^{T}\ell\_{t,k} $$

and

$$ L^\*(T)=\min_k L_k(T). $$

The regret becomes

$$ \boxed{R(T)=L_A(T)-L^\*(T)}. $$

This formulation accommodates more than correct/incorrect predictions. Losses might represent financial costs, prediction errors, or the penalty for choosing an action.

This is the conceptual bridge from learning from expert advice to broader online learning and, eventually, connections with conventional machine learning.

# Key takeaways table

| Concept                           | What to remember for the exam                                                                |
| --------------------------------- | -------------------------------------------------------------------------------------------- |
| Randomization                     | Makes the algorithm's prediction a random variable.                                          |
| Randomized Weighted Majority      | Predicts according to the relative weights of experts supporting each label.                 |
| Prediction probability            | $p_t=\text{weight supporting 1}/\text{total weight}$.                                    |
| Expected regret                   | Compare expected algorithm loss with the best expert's loss.                                 |
| Oblivious adversary               | Fixes the outcome sequence before play, independently of the algorithm's random realization. |
| Learning parameter $\epsilon$ | Controls the trade-off between the two terms in the regret bound.                            |
| Known horizon                     | Allows the choice $\epsilon=\sqrt{\ln D/T}$.                                             |
| Regret upper bound                | $O(\sqrt{T\ln D})$.                                                                      |
| Lower-bound insight               | In the worst case, square-root regret dependence on $T$ is unavoidable.                  |
| Optimality                        | The randomized algorithm achieves the optimal order in $T$ for fixed $D$.            |
| Doubling trick                    | Handles unknown $T$ using blocks of lengths $1,2,4,8,\ldots$.                        |
| General online learning           | Replace binary mistakes with cumulative losses.                                              |

# Critical formula sheet

| Quantity                    | Formula                                                           | Purpose                      |
| --------------------------- | ----------------------------------------------------------------- | ---------------------------- |
| Total expert weight         | $\displaystyle W_t=\sum\_{k=1}^D w\_{t,k}$                    | Normalization                |
| Probability of predicting 1 | $\displaystyle p_t=\frac{\sum\_{k:x\_{tk}=1}w\_{t,k}}{W_t}$   | Randomized decision          |
| Probability of predicting 0 | $1-p_t$                                                       | Complementary probability    |
| Weight update after error   | $w\_{t+1,k}=(1-\epsilon)w\_{t,k}$                             | Penalize a wrong expert      |
| Algorithm mistakes          | $\displaystyle M_A(T)=\sum\_{t=1}^T\mathbf1[\hat y_t\ne y_t]$ | Realized mistakes            |
| Best expert mistakes        | $\displaystyle M^\*(T)=\min_k M_k(T)$                         | Benchmark                    |
| Expected regret             | $\displaystyle \mathbb E[R(T)]=\mathbb E[M_A(T)]-M^\*(T)$     | Randomized performance       |
| Regret target               | $\displaystyle \mathbb E[R(T)]/T\to0$                         | Vanishing average regret     |
| Expected regret bound       | $\displaystyle \frac{\ln D}{\epsilon}+\epsilon M^\*(T)$       | Trade-off bound              |
| Worst-case simplification   | $\displaystyle \frac{\ln D}{\epsilon}+\epsilon T$             | Uses $M^\*(T)\le T$      |
| Parameter choice            | $\displaystyle \epsilon=\sqrt{\frac{\ln D}{T}}$               | Balance both terms           |
| Resulting bound             | $\displaystyle \mathbb E[R(T)]\le 2\sqrt{T\ln D}$             | Sublinear expected regret    |
| Lower-bound rate            | $\Omega(\sqrt T)$                                             | Worst-case limitation        |
| Doubling block length       | $L_j=2^{j-1}$                                                 | Handle unknown horizon       |
| Parameter per block         | $\displaystyle \epsilon_j=\sqrt{\frac{\ln D}{L_j}}$           | Tune each block              |
| General cumulative loss     | $\displaystyle L_k(T)=\sum\_{t=1}^T\ell\_{t,k}$               | Beyond binary mistakes       |
| General regret              | $\displaystyle R(T)=L_A(T)-\min_k L_k(T)$                     | General benchmark comparison |

Notation note: $\ln$ denotes the natural logarithm. The displayed regret bounds use the lecture's parameterization and appropriate parameter-range conditions.

# Common exam traps

Trap 1: Confusing deterministic and randomized predictions

Deterministic Weighted Majority follows the side with greater weight. Randomized Weighted Majority samples from the weighted distribution.

Trap 2: Confusing regret with mistakes

An algorithm can make many mistakes but still have low regret if the best expert also makes many mistakes.

Trap 3: Assuming the best expert is perfect

The best expert minimizes mistakes in hindsight. It need not make zero mistakes.

Trap 4: Thinking sublinear means constant regret

Square-root regret grows with $T$, but its ratio to $T$ tends to zero.

Trap 5: Forgetting the horizon assumption

The direct parameter choice depends on $T$. The doubling trick addresses this when $T$ is unknown.

# Five-minute revision summary

Memorize this chain:

$$ \boxed{ \begin{gathered} \text{Deterministic Weighted Majority}\\\ \downarrow\\\ \text{Randomize the prediction}\\\ \downarrow\\\ \text{Analyze expected regret}\\\ \downarrow\\\ \epsilon=\sqrt{\ln D/T}\\\ \downarrow\\\ \mathbb E[R(T)]\le 2\sqrt{T\ln D}\\\ \downarrow\\\ \text{Sublinear expected regret}\\\ \downarrow\\\ \text{Doubling trick if }T\text{ is unknown} \end{gathered} } $$

The deepest lesson from Week 2 is that randomization is not merely a way to introduce unpredictability. It changes what performance guarantees are achievable against an adversary. The square-root regret bound is the mathematical evidence.

For revision, I would next practise numerical questions on prediction probabilities, selecting $\epsilon$, evaluating regret bounds, and identifying the correct adversary model. Those are the natural places for an examiner to turn these ideas into calculations.
