Your suspicion is pointing at exactly the part of the argument that deserves scrutiny: the algorithm repeatedly throws away its weights, so why should its regret improve rather than accumulate?

The key is that the doubling trick does not make the regret from earlier blocks disappear. Instead, it arranges the blocks so that the regret accumulated across all the restarts grows only as $O(\sqrt{T})$, up to a factor involving the number of experts.

There is also a subtle issue in the lecture's notation and derivation. Let's separate the correct idea from the parts that need fixing.

## 1. What is the algorithm doing?

Suppose we have $d$ experts, and randomized weighted majority (RWM) has the finite-horizon regret bound

$$ R(n)\leq 2\sqrt{n\ln d}. $$

This bound applies when we know the horizon $n$ and choose the learning parameter appropriately, for example $\epsilon=\sqrt{\ln d/n}$.

But we want to run the algorithm indefinitely, without knowing when someone will ask us to evaluate its regret.

The doubling trick runs independent epochs:

1

Epoch 1

1 round, then restart

2

Epoch 2

2 rounds, then restart

4

Epoch 3

4 rounds, then restart

8

Epoch 4

8 rounds, then restart

The epoch lengths keep doubling: 1, 2, 4, 8, 16, …

Every epoch starts with fresh weights and uses a learning rate suited to that epoch's length. So yes, the algorithm forgets its previous weights. That is intentional: each epoch can independently use the finite-horizon guarantee.

## 2. Why don't the restarts destroy sublinearity?

The crucial observation is that the regret bound for an epoch grows as the square root of its length, not linearly with its length.

For an epoch of length $n$,

$$ R\_{\text{epoch}} \leq 2\sqrt{n\ln d}. $$

Thus the regret bounds for the successive epochs are:

| Epoch length | Regret upper bound    |
| ------------ | --------------------- |
| $1$      | $2\sqrt{\ln d}$   |
| $2$      | $2\sqrt{2\ln d}$  |
| $4$      | $2\sqrt{4\ln d}$  |
| $8$      | $2\sqrt{8\ln d}$  |
| $16$     | $2\sqrt{16\ln d}$ |

The regret from each epoch still accumulates. But the cost of each successive epoch grows geometrically, with ratio $\sqrt{2}$, rather than growing by a fixed amount per round.

The total bound through the end of epoch $k$ is

$$ \begin{aligned} R_T &\leq 2\sqrt{\ln d} \left(1+\sqrt{2}+2+2\sqrt{2}+\cdots\right)\\\ &=2\sqrt{\ln d}\sum\_{i=0}^{k}\sqrt{2^i}. \end{aligned} $$

This is a geometric series with ratio $\sqrt{2}$. Its sum is dominated by its last term:

$$ \sum\_{i=0}^{k}2^{i/2}=O(2^{k/2}). $$

And because the epoch lengths double, $2^k$ is at most on the order of $T$. Consequently,

$$ \boxed{R_T=O(\sqrt{T\ln d})}. $$

The restarts do not eliminate regret. They ensure that the sum of the per-epoch regret guarantees remains sublinear.

## 3. The subtle point: regret is measured against one global expert

There is a mathematical detail here that makes the argument work.

Let $L\_{\text{alg},j}$ be the algorithm's loss in epoch $j$, and $L\_{i,j}$ be expert $i$'s loss in that epoch.

The regret of epoch $j$, measured against that epoch's best expert, is

$$ r_j=L\_{\text{alg},j}-\min_i L\_{i,j}. $$

Summing across epochs gives

$$ \sum_j r_j =\sum_j L\_{\text{alg},j}-\sum_j\min_i L\_{i,j}. $$

But the best expert can differ between epochs. The key inequality is

$$ \sum_j\min_i L\_{i,j} \leq \min_i\sum_j L\_{i,j}. $$

Why? If you allow yourself to pick the best expert separately for each epoch, you can do at least as well as being forced to use one expert for the entire game.

Therefore,

$$ \boxed{R_T\leq \sum_j r_j}. $$

This means that bounding the regret of each epoch separately really does give an upper bound on the regret against the single best expert in hindsight over all $T$ rounds.

## 4. Where is the actual flaw in the lecture?

The central logic is sound, but there are two details worth correcting.

First, the bound must also cover a partially completed epoch.

Suppose you stop at round $T=13$, during the epoch of length $8$. You cannot simply claim that the current epoch has completed all eight rounds. Instead, use the regret guarantee for its prefix, or upper-bound that prefix by the guarantee for the full epoch.

Since the full-epoch bound is $2\sqrt{8\ln d}$, this is still valid. The geometric-series argument continues to work.

Second, the final rate should be stated carefully.

For $d$ experts, the derived rate is

$$ \boxed{R_T=O(\sqrt{T\ln d})}, $$

not necessarily $O(\sqrt{T\ln T})$. The logarithm is of the number of experts, $d$, not the time horizon $T$. If $d$ is fixed, $\sqrt{\ln d}$ is just a constant, and the regret is $O(\sqrt T)$.

## 5. Your intuition, sharpened

Think of the resets as paying a fee for starting over. With equal-sized blocks, you pay that fee once every $K$ rounds, so the total fee grows linearly with time. With doubling-sized blocks, you pay the fee less and less frequently.

The algorithm does forget what it learned. But it forgets at carefully spaced intervals, and the finite-horizon guarantee ensures that each interval's regret is controlled.

Bottom line: There is no fundamental flaw in the doubling trick. The proof works because the sum of square-root-sized regret bounds over geometrically increasing epochs is itself dominated by the square root of the total time. Your concern about discarded weights is valid, but the analysis explicitly pays for each restart rather than pretending the past never happened.

The “magic” is a mathematical property called geometric growth, combined with the fact that the regret grows as the square root of an episode's length.

But there is a deeper way to see it. Let's compare the three approaches you described.

## 1. The three strategies

For randomized weighted majority, suppose the finite-horizon regret bound is

$$ R(n)\leq 2\sqrt{n\ln d}. $$

Here $n$ is the number of rounds and $d$ is the number of experts.

### Strategy A: One infinite episode

If we knew the total horizon $T$, we could choose

$$ \epsilon=\sqrt{\frac{\ln d}{T}}. $$

Then the regret is $O(\sqrt{T\ln d})$, which is sublinear.

But for an infinite run, $T=\infty$ would force $\epsilon=0$, which is not a useful learning rate. We need an approach that works without knowing $T$.

### Strategy B: Equal-sized episodes

Choose a fixed episode length $K$. Each episode has regret at most $2\sqrt{K\ln d}$.

By time $T$, there are approximately $T/K$ episodes. Therefore,

$$ \begin{aligned} R_T &\leq \frac{T}{K}\\,2\sqrt{K\ln d}\\\ &=2T\sqrt{\frac{\ln d}{K}}. \end{aligned} $$

Divide by $T$:

$$ \boxed{\frac{R_T}{T} \leq 2\sqrt{\frac{\ln d}{K}}}. $$

For fixed $K$, this upper bound is a constant, not something that tends to zero. The bound does not establish sublinear regret.

### Strategy C: Doubling-sized episodes

Now use episode lengths $1,2,4,8,16,\ldots$.

The regret bounds become

$$ 2\sqrt{\ln d}\left( 1+\sqrt2+2+2\sqrt2+\cdots \right). $$

The important change is that the number of episodes grows logarithmically with time, and the sum of their regret bounds is dominated by the largest episode.

Thus,

$$ R_T=O(\sqrt{T\ln d}), $$

and so

$$ \boxed{\frac{R_T}{T} =O\left(\sqrt{\frac{\ln d}{T}}\right) \longrightarrow 0}. $$

That is sublinear regret.

## 2. What property caused the magic?

There are two interacting properties.

Property 1: Regret grows sublinearly within an episode

An episode of length $n$ incurs regret proportional to $\sqrt n$, rather than $n$. Doubling the episode length only multiplies its regret bound by $\sqrt2$.

Property 2: Geometric episode lengths

The total regret is a geometric series with ratio $\sqrt2$, which is greater than 1. Such a series is dominated by its last term, so the accumulated regret stays on the order of $\sqrt T$, rather than $T$.

The second property is what makes the restart schedule effective. The first property makes that schedule useful.

## 3. The general mathematical principle

Suppose the regret for an episode of length $n$ is bounded by

$$ R(n)\leq Cn^\alpha, \qquad 0<\alpha<1, $$

where $C$ is a constant.

With doubling episode lengths, the total regret through episode $k$ is bounded by

$$ \begin{aligned} R_T &\leq C\sum\_{i=0}^{k} (2^i)^\alpha\\\ &=C\sum\_{i=0}^{k}2^{i\alpha}\\\ &=C\frac{2^{(k+1)\alpha}-1}{2^\alpha-1}. \end{aligned} $$

Since $2^k$ is on the order of $T$,

$$ \boxed{R_T=O(T^\alpha)}. $$

Consequently,

$$ \boxed{\frac{R_T}{T}=O(T^{\alpha-1})\to0}. $$

This is the general principle: if finite-horizon regret grows like a power $T^\alpha$, with $\alpha<1$, doubling the horizon at each restart preserves that sublinear growth rate.

For weighted majority, $\alpha=\tfrac12$, giving the square-root regret bound.

## 4. Why equal-sized episodes fail, in one line

With fixed episode size $K$, the number of episodes is proportional to $T$, so you repeatedly pay the same regret bound:

$$ \underbrace{O(\sqrt K)+O(\sqrt K)+\cdots+O(\sqrt K)}\_{\text{about }T/K\text{ episodes}} =O(T). $$

With doubling-sized episodes, the number of episodes is only $O(\log T)$, and their costs form a geometric series dominated by the final episode:

$$ O(1)+O(\sqrt2)+O(2)+\cdots+O(\sqrt T) =O(\sqrt T). $$

The crucial difference is not merely that the episodes get bigger. It is that the cost of restarting is spread over exponentially increasing amounts of time.

One final nuance: the doubling trick is not the only possible schedule. It is a particularly clean way to turn a finite-horizon guarantee into an anytime algorithm, provided the per-episode regret grows sublinearly.
