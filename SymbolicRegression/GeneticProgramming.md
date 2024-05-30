---
title: Genetic Programming
author:
  - name: Prof. Fabrício Olivetti de França
institute: Federal University of ABC
logo:
  - name: figs/logotipo-ufabc-abaixo.eps
  - name: figs/hal.png
cover: figs/cover.png
date: 05 February, 2024
---

## Evolution
\justifying

\notebox{The reasonable man adapts himself to the world; the unreasonable one persists in trying to adapt the world to himself. Therefore all progress depends on the unreasonable man.}

-- George Bernard Shaw, *Maxims for Revolutionists*

# Search and Optimization

## Real-world problems
\justifying

Some real-world problems are hard to solve:

- There are many possible solutions, we cannot enumerate them all
- We cannot formalize the problem description, thus requiring simplifications
- The evaluation criteria can be noisy, or change with time
- There are many constraints associated with the main problem

## Search and Optimization Problems
\justifying

A problem can be formalized as either a **search problem** or an **optimization problem**.

## Search Problem
\justifying

A **search problem** is when given a set of candidate solutions $S$ and a property $P: S \rightarrow \{T, F\}$, we have to find $s$ such that $s \in S, P(s)$.


## SAT
\justifying

The Boolean Satisfability problem (SAT) is described as:

\notebox{Given a boolean function $f(x)$, assign \textbf{true} or \textbf{false} values to each $x_i$ such that the function evaluates to \textbf{true}.}

## SAT
\justifying

For example:

\begin{empheq}[box=\mybox]{align*}
f(x) = (x_1 \lor x_3 \lor \lnot{x_4}) \land (x_2 \lor \lnot{x_3})
\end{empheq}

## SAT
\justifying

To make things simpler, we can use $0$ and $1$ to represent false and true, respectivelly.

With $4$ variables, our set $S$ has $|S| = 2^4 = 16$ candidate solutions.

|$x_1$|$x_2$|$x_3$|$x_4$|
|-----|-----|-----|-----|
|  0  |  0  |  0  |  0  |
|  0  |  0  |  0  |  1  |
|  0  |  0  |  1  |  0  |
|  0  |  0  |  1  |  1  |

...

## SAT
\justifying

As the number of variables grow, the cardinality of $S$ grows quite fast:

![](figs/SAT.pdf){ width=300px }

## SAT

| $n$ | $2^n$ |
|-----|-------|
|  1  |   2   |
|  5  |  32   |
| 10  | 1024  |
| 100 | 1.2e30|
|1000 |1.1e301|

## Optimization problem
\justifying

An **optimization problem** is when you have a set of candidate solutions $X$ and a criteria function $F : X \rightarrow \mathbb{R}$ and you want to find $x \in X, x \in argmax_{y \in X} f(y)$, for maximization problems and $x \in X, x \in argmin_{y \in X} f(y)$ for minimization problems.

## Optimization problem
\justifying

\notebox{\textbf{maximization problems} have the objective of finding one of possibly many solutions that has a maximum value for the criteria function.}

\notebox{\textbf{minimization problems} have the objective of finding one of possibly many solutions that has a minimum value for the criteria function.}

## Nonlinear Optimization
\justifying

\begin{empheq}[box=\mybox]{align*}
G2(x) &= \left|\frac{\sum_{i=1}^n{cos^4(x_i)} - \prod_{i=1}^{n}{cos^2(x_i)}}{\sqrt{\sum_{i=1}^n{ix_i^2}}}\right| \\
\text{subject to} \\
\prod_{i=1}^n{x_i \geq 0.75} \\
\sum_{i=1}^n{x_i} \leq 7.5n \\
0 \leq x_i \leq 10, 1 \leq i \leq n
\end{empheq}

## Nonlinear Optimization
\justifying

Conceptually there are an unlimited number of candidate solutions. But, when solving using floating point numbers in a computer, we have a (large but) finite number of solutions.

## Nonlinear Optimization
\justifying

Assuming we can represent up till $6$ decimal places, we would have $10.000.000$ distinct values for each variable $x_i$.

So we have $10.000.000^n = 10^{7n}$ candidate solutions.

## Search and Optimization Problems
\justifying

The main difference between search and optimization problem is that when you find a solution $s, P(s)$, we can stop the search and return the solution.

With an optimization problem, we have to check all possible solutions to be sure it is the best solution.

# Basic Concepts

## Search Space

\notebox{\textbf{search space} is the set of all candidate solutions.}

## Solution representation

\notebox{\textbf{solution representation} is how we conveniently represent a solution to our problem.}

## Objective-function

\notebox{an \textbf{objective-function} is a function that maps a candidate solution to a (usually real) value measuring the quality of that solution. It can be a maximization or minimization function.}

## Neighborhood

\notebox{the \textbf{neighborhood} of a solution $s$ is the set of all solutions close to $s$.}

## Neighborhood
\justifying

The neighborhood can be defined using a distance measure or a function that projects a solution to a power set.

## Neighborhood
\justifying

For example, for the nonlinear optimization problem, the representation is a vector $x \in \mathbb{R}^d$ and the neighborhood can be defined through a distance measure $d$:

\begin{empheq}[box=\mybox]{align*}
d(x^{(1)}, x^{(2)}) &= \sqrt{\sum_{i=1}^{d}{(x^{(1)}_i - x^{(2)}_i)^2}} \\
\mathcal{N}(X) &= \left\{y \in S \mid d(x, y) \leq \epsilon\right\}
\end{empheq}

## Neighborhood
\justifying

For the SAT problem, we can represent the solution as a binary vector and the neighborhood set is the function $\mathcal{N} : S \rightarrow 2^S$ that enumerates all set of solutions that swaps the value of two elements of the current solution.

## Neighborhood
\justifying

Fundamentally, a function $f : S \rightarrow 2^S$ is equivalent to $f : S \rightarrow S \rightarrow 2$.

That, uncurrying, is the same as $f : S \times S \rightarrow 2$.

In other words, it is a function that takes two solutions and returns $0$ or $1$ depending whether they are neighbors or not.

## Local search
\justifying

**Local search** algorithms start from an initial solution and iteratively walks through the neighborhood until it reaches a point without a better neighbor.

This point is called **local optimum**.

## Local search
\justifying

In some situations, we can guarantee that the **local optimum** is the best solution (or, **global optimum**).

For example, when maximizing $f(x) = -x^2$, iteratively walking through the neighborhood will always reach the global optima (under certain conditions).

## Local search
\justifying

But in many problems, depending on the amount of local optima, this procedure can get stuck at a subpar solution.

\centering
![](figs/hill-climbing.eps)

## Hill-climbing {.fragile}
\justifying

The **hill-climbing** algorithm or **steepest ascent**, repeats iteratively:

```{.haskell frame=lines framerule=2pt linenos=true fontsize=\footnotesize baselinestretch=0.8}
hill-climbing solution =
  best-neighbor = best (neighbors solution)
  if best-neighbor `better-than` solution
    then hill-climbing best-neighbor
    else return solution
```

## Random Search {.fragile}
\justifying

Consider now an algorithm that keeps generating and evaluating random solutions.

This is called a **random search**:

```{.haskell frame=lines framerule=2pt linenos=true fontsize=\footnotesize baselinestretch=0.8}
random-search =
  solution = random-solution
  return solution : random-search

get-best random-search
```

## Complete and Optimal searches
\justifying

A search algorithm is called **complete** if it guarantees to return a feasible solution (i.e., any $s, P(s)$).

A search algorithm is called **optimal** if it always return the global optimum.

## Complete and Optimal searches
\justifying

The Hill-climbing is neither complete nor optimal since it cannot guarantee any of these properties.

A random search is both complete and optimal since, given an infinite amount of time, it will eventually find the best solution.

## Simulated Annealing
\justifying

Since we don't want to wait an infinite amount of time, we may try something in between Hill-climbing and random search.

**Simulated annealing** is one of such algorithm. It is essentialy a hill-climbing algorithm but the decision of whether to replace the current solution with one of the neighbors is not deterministic.

## Simulated Annealing {.fragile}
\justifying

```{.haskell frame=lines framerule=2pt linenos=true fontsize=\footnotesize baselinestretch=0.8}
simulated-annealing solution T =
 | T <= eps = return solution
 | otherwise =
    neighbor = random-neighbor solution
    next-solution =
    if neighbor `isBetterThan` solution
      || random <= exp (eval neighbor - eval solution)/T
      then neighbor
      else solution
    return simulated-annealing next-solution (shrink T)
```

## Simulated Annealing
\justifying

The main idea of SA is to pick a random neighbor $s'$ and accept it (i.e., replace current solution) if:

- This neighbor is better than the current solution or
- Accept with a probability of $e^{\frac{f(s') - f(s)}{T}}$

## Simulated Annealing
\justifying

Assuming maximization, if $f(s') = f(s)$ it will always replace as it has the same objective-function value.

If $f(s') < f(s)$ it will make the probability goes toward zero the higher the difference between both solutions. The value of $T$ will determine how much worse we are willing to accept.

## Simulated Annealing
\justifying

\begin{tikzpicture}
\begin{axis}[legend pos=outer north east, domain=-10:0]
\addplot[very thick,black,legend entry=$0.1$] {exp(x/0.1)};
\addplot[very thick,black,legend entry=$1$] {exp(x/1)};
\addplot[very thick,black,legend entry=$10$] {exp(x/10)};
\addplot[very thick,black,legend entry=$100$] {exp(x/100)};
\end{axis}
\end{tikzpicture}

## Simulated Annealing
\justifying

Starting with a large value of $T$, SA will behave similarly to a random search, accepting almost any solution.

As the value of $T$ is reduced at every iteration, it will start to behave more like a hill climbing, as it will be less likely to accept a worse soluion.

# Heuristic

## Heuristic
\justifying

\notebox{\textbf{Heuristic}, from the greek \emph{to find} or \emph{discover}, are tecniques developed to find a solution to a problem without any guarantees.}

## Heuristic
\justifying

The main goal of a heuristic method is to find an approximate solution as efficiently as possible.

## Heuristic
\justifying

George Pólya[^1] enumerates some tips to create a heuristic to solve a problem:

- If you cannot understand the problem, draw a diagram representing it.
- If you cannot reach a solution from an initial state, try reaching the initial state from the solution.
- If the problem is abstract, try creating a concrete example.
- Try to solve a less restrictive problem first.

[^1]: How to Solve it, 1945

## Greedy Heuristic
\justifying

\notebox{A \textbf{greedy heuristic} tries to iteratively build a single solution by maximizing the instant reward.}

## Greedy Heuristic
\justifying

A possible greedy heuristic for the optimization problem is to fix every variable to a starting value, and optimize a single variable at every iteration.

## Populational Heuristic
\justifying

The local search and greedy heuristics so far worked with a single solution.
Their main limitation was to reach the closest optima to the starting point.

\centering
![](figs/hill-climbing.eps)

## Populational Heuristic
\justifying

What if we start at multiple points in parallel?

\centering
![](figs/hill-climbing.eps)

## Populational Heuristic
\justifying

By doing so we can reach multiple local optima at once, increasing our chances of finding the global optima.

We can explore this idea even further by creating a competition among the different solutions and focus the search only on the promising regions.

## Populational Heuristic
\justifying

Also, we can try to combine parts of different good solutions with the expectation of creating a better solution.

## Populational Heuristic
\justifying

The main idea of a populational heuristic is to find a balance between **exploration** and **exploitation**.

## Populational Heuristic
\justifying

\notebox{ \textbf{exploration} is the act of exploring the whole search space. Trying to find new promising regions.}

## Populational Heuristic
\justifying


\notebox{\textbf{exploitation} is the act of exploring only the local neighborhood to find the best solution in this smallr region.}

## Populational Heuristic
\justifying

Both are equally important as we need to find promising regions while extracting the best from the current best regions.

## Populational Heuristic
\justifying

Let us suppose we create $30$ random solutions for the SAT problem.

We can do that by sampling random bits with a probability $p=0.5$ of sampling a $1$.

## Populational Heuristic
\justifying

Next, we randomly select with replacement $30$ solutions from this population with a probability proportional to the objective-function.

## Populational Heuristic
\justifying

From this selection, we will apply a random perturbation and repeat these procedures with this new population.

## Populational Heuristic
\justifying

For a SAT problem with $91$ terms and $20$ variables, in $30$ executions with $200$ solutions and a maximum of $1000$ iterations, this simple procedure will

- Reach the goal in $24$ of the executions
- Reach a solution on average after $272$ iterations

## Populational Heuristic  {.fragile}
\justifying

An abstract description of this algorithm is:


```{.haskell frame=lines framerule=2pt linenos=true fontsize=\footnotesize baselinestretch=0.8}
population-heurisitc =
  pop = random-population
  f   = eval pop
  while not done
    selection = selectWithReplacement pop f
    pop       = perturb selection
  return pop
```

One example of populational heuristic is the **evolutionary algorithm**.

# Evolutionary Algorithms

## Evolution
\justifying

Evolution is a natural process in which a species adapts itself successively with the objective of:

- survival
- ecological balance
- diversity

## Natural Selection
\justifying

Natural Selection was proposed by Charles Darwin to explain why some characteristics become very common while others disappear.

## Natural Selection
\justifying

1. More offsprings are produced than necessary
2. Characteristics of an individual defines its probability of survival
3. Characteristics are hereditary

## Evolution
\justifying

Let's illustrate with the constant fight for survivor of rabbits in a territory filled with foxes.

## Evolution
\justifying

In a population of rabbits, some are more clever and some are faster than others.

Those have more chance of survival and produce __more rabbits__.

## Evolution
\justifying

With fast rabbits reproducing with clever rabbits, new variants of rabbits appear:

- fast and dumb
- slow and clever
- slow and dumb
- fast and clever

## Evolution
\justifying

Nature sometimes throw a *crazy rabbit* in this population with some variation in the genes.

The offsprings are not exact copy of their parents, but random variations (remember the regression to the mean).

![](figs/bioinsp32.jpg){ width=200px }

## Evolution
\justifying

Throughout the generations, hopefully the population of rabbits become faster and more clever than the initial population.

But the foxes also evolve...

## Evolutionary Algorithms {.fragile}
\justifying

```{.haskell frame=lines framerule=2pt linenos=true fontsize=\footnotesize baselinestretch=0.8}
p = initial-population
while not done
  parents    = select-from p
  offsprings = recombine-from parents
  xmen       = mutate offsprings
  p          = replace-from (p + xmen)
return p
```

## Population
\justifying

Each individual of the population represents a solution represented as seen fit.

In evolutionary algorithms this representation is called **chromosome**.

## Selection
\justifying

At this stage, the soon-to-be parents are selected from the population with a probability proportional to their fitness.

The idea is that the fittest have a higher chance of reproducing and passing their characteristics to their offsprings.

## Recombination
\justifying

Recombination simulates the reproduction of individuals of the same species where the exchange of genetic material happens.

![](figs/bioinsp33.jpg){ width=200px }

## Recombination {.fragile}
\justifying

Assuming a vectorial representation:

```haskell
parent1 = [3 0 2 | 5 2 1]
parent2 = [5 1 2 | 5 0 3]

child1 = [3 0 2 5 0 3]
child2 = [5 1 2 5 2 1]
```

## Mutation
\justifying

Mutation promotes novelty in our current population.

It prevents that every individual is the same.

## Mutation {.fragile}
\justifying

Mutation in this algorithm is a random perturbation in the solution representation:

```haskell
child   = [3 0 *2* 5 2 1]
xman = [3 0 *4* 5 2 1]
```
## Replacement
\justifying

Once we have a population of offsprings, we will replace the current population with either the offsprings or a mix of the current population and their offsprings.

## Replacement
\justifying

Whenever we need to make a decision based on how fit an individual is, we use the objective-function or an adaptation of that to calculate the **fitness** of an individual.

Notice that conceptually the fitness is a maximization objective, but it is sometimes used as minimization for some problems.

## Taxonomy
\justifying

- __Genetic Algorithms__ : proposed by Holland with the main goal of studying the adaptation phenomena.
- __Evolution Strategies__: introduced by Rechenberg to solve parameter optimization of nonlinear functions.
- __Evolutionary Programming__: represents a program as a finite state machine, proposed by Fogel at al.
- __Genetic Programming__: main goal of evolving computer programs, proposed by Koza.

## Evolutionary Algorithms
\justifying

Evolutionary Algorithms were proposed by different researchers in different forms, but with some common properties, they all

- Work with the idea of population of solutions
- Use selective pressure, with a likelihood of survival proportional to the fitness
- Measure the quality of each solution with a maximization objective-function
- Combine set of solutions thus expanding the local neighborhood
- Perturb selected solutions at random

## Evolutionary Algorithms
\justifying

This generates two fundamental forces in this class of algorithms:

- **Variation Operators:** promote diversity and search for new solutions (exploration).
- **Selective Pressure:** promote maintenance of good quality solutions (exploitation).

## Terminology

- **Individual:** a solution represented by a *chromosome*.
- **Chromosome:** the computational representation of a solution, also known as **genotype**.
- **Phenotype:** the decoded representation.
- **Fitness:** a maximization objective-function that measures the quality of an individual.
- **Population:** a bag of individuals.
- **Crossover:** or **recombination**, a function that combines the information of two or more individuals.
- **Mutation:** a function that randomly changes one individual.

# Selection Strategies

## Selection proportional to fitness
\justifying

In this strategy, assuming $f_i$ the fitness of the $i$-th individual, the probability of selecting it as a parent is (assuming non-negative fitness):

\begin{empheq}[box=\mybox]{align*}
p_i = \frac{f_i}{\sum_j{f_j}}
\end{empheq}

So the probability is proportional to the absolute value of the fitness.

## Selection proportional to fitness
\justifying

This strategy has some drawbacks:

- Individuals with much higher fitness may end up dominating the selection causing a **premature convergence**.
- When the fitness values are too close to each other, there is no selective pressure and the choice is almost uniformly at random.
- The probability of choosing an individual may be sensitive to small variations in fitness.

## Selection proportional to fitness
\justifying

![Eiben, Agoston E., and James E. Smith. Introduction to evolutionary computing. Springer-Verlag Berlin Heidelberg, 2015.](figs/eiben319.png){width=200px}

## Selection proportional to fitness
\justifying

We can alleviate some of these problems with the **sigma scaling** proposed by Goldberg:

\begin{empheq}[box=\mybox]{align*}
f'_i &= max(f_i - (\bar{f} - c \sigma_f), 0.0) \\
p_i &= \frac{f'_i}{\sum_j{f'_j}}
\end{empheq}

where $\bar{f}, \sigma_f$ are the mean and standard deviation fitness of the population and $c$ is a scaling constant, usually set to $2$.

## Rank Selection
\justifying

Another strategy is to use the **rank** of the solutions instead of the fitness.

For example, the fitness:

\centering
$[0.1, 19.3, 1.4, 0.05]$

Would be mapped to:

\centering
$[2, 4, 3, 1]$

## Rank Selection
\justifying

To calculate the selection probability based on rank, we can use a linear or exponential scale:

\begin{empheq}[box=\mybox]{align*}
rank_{linear}(i) &= \frac{2-s}{\mu} + \frac{2(i-1)(s-1)}{\mu (\mu-1)} \\
rank_{exp}(i) &= \frac{1 - e^{-i}}{\sum_j{1 - e^{-j}}}
\end{empheq}

where $i$ is the rank value, $1 < s < 2$ the scaling factor, $\mu$ is the highest observed rank.

## Roulette Wheel Selection{.fragile}
\justifying

The roulette selection assigns a region of a wheel to each individual, pick a random value, and returns the individual at that spot of the wheel.

\begin{empheq}[box=\mybox]{align*}
P(x_i) = \frac{\operatorname{fitness}(x_i)}{\sum_j{\operatorname{fitness}(x_j)}}
\end{empheq}

```{.haskell frame=lines framerule=2pt linenos=true fontsize=\footnotesize baselinestretch=0.8}
spin fs = do
  r     = random (0, 1)
  wheel = cumsum(probability fs)
  return (firstOf (>r) wheel)
```

## Roulette Wheel Selection{.fragile}
\justifying

If the fitness of our individuals are $[324, 1, 100, 289]$

The probability for each individual is $[0.454, 0.001, 0.140, 0.405]$.

## Roulette Wheel Selection{.fragile}
\justifying

Next we calculate the cummulative sum of these probabiilities, defining the slice of the wheel:

$[0.454, 0.455, 0.595, 1.000]$

## Roulette Wheel Selection{.fragile}
\justifying

Now we choose a random value $0 \leq r \leq 1$ and check which slice this value belongs to. The individual within this slice is chosen as one of the parents for reproduction.

$[0.454, 0.455, 0.595, 1.000]$

For $r = 0.3$ we choose the first individual.

For $r = 0.4541$ we choose the second individual.

## SUS
\justifying

The main problem with this approach is that by spinning the wheel multiple times we may not generate a representative sample of our population.

We can expand this idea by spinning the wheel with multiple arrows and sample multiple individuals at once. This is known as **stochastic universal sampling** (SUS)

## SUS {.fragile}

```{.haskell frame=lines framerule=2pt linenos=true fontsize=\footnotesize baselinestretch=0.8}
sus n fs = do
  r       = random(0, 1/n)
  wheel   = cumsum(probability(fs))
  choices = for [1..n] (\i -> firstOf (>(r*i)) wheel)
  return choices
```

## SUS {.fragile}
\justifying

This strategy guarantees that the number of times the $i$-th individual is selected is at least the integral part of $n \cdot P(i)$.

## Tournament Selection
\justifying

The tournament selection simply samples $k$ individuals of the population and returns the best among them.

This is a local selection strategy, since it only uses the knowledge about the $k$ sampled individuals as opposed to the roulette wheel that requires knowing the fitness of the entire population.

## Tournament Selection
\justifying

An important characteristic of this strategy is that it is invariant to translation and transposition. The choice of fitness does not affect the results of this method.

## Tournament Selection
\justifying

The probability that an individual is selected depends of:

- Its rank in the population
- The size $k$ of the tournament (the higher the size, the higher the bias to above average individuals)
- If the individuals are selected with or without replacement (without replacement, the $k-1$ worse will never be selected)

## Tournament Selection  {.fragile}
\justifying

```{.haskell frame=lines framerule=2pt linenos=true fontsize=\footnotesize baselinestretch=0.8}
tournament k pop = do
  competitors = sampleWithReplacement(k, pop)
  return (best competitors)
```

## Lexicase Selection  {.fragile}
\justifying

Most selection strategies so far compare individuals using an aggregated score (i.e., the fitness).

In some situations, we can evaluate the individual locally.

For example, in regression we can calculate the squared residue for a single point or the sum of squared residues for a selection of the sample.

## Lexicase Selection  {.fragile}
\justifying

Evaluating the aggregated value we stimulate **generalists**. But during the evolution process we may want to combine two or more **specialists**.

## Lexicase Selection  {.fragile}
\justifying

Assuming maximization (fitness) and $xy$ is the list of tuples $(x,y)$ from our sample:

```{.haskell frame=lines framerule=2pt linenos=true fontsize=\footnotesize baselinestretch=0.8}
lexicase pop xy =
  pool  = pop
  cases = shuffle xy
  return (while-loop pool cases)

while-loop pool (case:cases)
  | sizeOf pool  == 1 = return (head pool)
  | sizeOf cases == 0 = return (random pool)
  | otherwise =
     best  = max (partial-fit case) pool
     pool' = filter (==best . partial-fit case) pool
     while-loop pool' cases
```
## Lexicase Selection  {.fragile}
\justifying

When our fitness is continuous and we want some tolerance between two very similar individuals, we can use the $\epsilon$-lexicase selection.

In this version, we simply replace the comparison with the best value with a comparison to whether the difference between the best value and the current value is within an $\epsilon$ radius.

## Replacement
\justifying

Evolutionary Algorithms often works with one of two popular models: **generational** and **stationary**.

## Generational
\justifying

\notebox{Generational model in a population of size $\mu$ we choose $\mu$ parents and generate $\lambda = \mu$ offsprings that entirely replace the parent population.}

## Stationary
\justifying

\notebox{In the stationary model, we generate $\lambda < \mu$ offsprings to replace some of the current solutions in the population. A special case is when $\lambda = 1$.}

## Replacement
\justifying

When using the stationary model, we have to select which individuals will survive and which will be replaced.

## Fitness based replacement
\justifying

In the fitness based replacement, we use one of the selection strategies to select the next generation.

## Replace the worst
\justifying

In this strategy we pick $n$ worse individuals of the current population and replace with the offsprings.

## Elitism
\justifying

In this strategy, we keep the best $n$ individuals in the population and use another strategy to select the remainders.

## Multi-objective
\justifying

Sometimes we want to optimize more than one objective and they are conflicting with each other (if we improve one, we make the other worse).

For these situations we can apply a Multi-objective version of the evolutionary algorithms.

## Multi-objective
\justifying

The main difference is in the selection and reproduction operators in which the comparison between individuals are made through the dominance operator instead of equality.

## Multi-objective
\justifying

In multi-objective optimization we evaluate the individuals using $m > 1$ objectives. As such, we have to redefine how we compare two solutions.

## Multi-objective
\justifying

Instead of saying one solution is better than another, we say that one solution dominates another (denoted by $\mathbf{f}(\mathbf{x_1}) \prec \mathbf{f}(\mathbf{x_2})$) if both:

\begin{empheq}[box=\mybox]{align*}
    \forall i \in \{1 \ldots m\} : f_i(\mathbf{x_1}) \leq f_i(\mathbf{x_2}), \\
    \exists i \in \{1 \ldots m\} : f_i(\mathbf{x_1}) \neq f_i(\mathbf{x_2}),
\end{empheq}

are true considering a minimization problem.

## Multi-objective
\justifying

In short, this means that a certain solution dominates the other if it is equal or better in every objective and better in at least one of them.

## Multi-objective
\justifying

The Pareto optimal set is defined as the set of optimal solutions that are not dominated by any other:

\begin{empheq}[box=\mybox]{align*}
    \mathbb{P} = \{ \mathbf{x}^* \in \Omega \mid \nexists \mathbf{x} \in \Omega : \mathbf{f}(\mathbf{x}) \prec \mathbf{f}(\mathbf{x^*})\}.
\end{empheq}

## Multi-objective
\justifying

And the Pareto front is represented as the image of this set:

\begin{empheq}[box=\mybox]{align*}
    \mathcal{F} = \{ \mathbf{f}(\mathbf{x}) : \mathbf{x} \in \mathbb{P}\}.
\end{empheq}

## Multi-objective
\justifying

A Multi-objective approach for evolutionary algorithms is the Fast Non-dominated sorting algorithm (NSGA-II)[^2] together with the Crowding Distance.

[^2]: Deb, Kalyanmoy, et al. "A fast and elitist multiobjective genetic algorithm: NSGA-II." IEEE transactions on evolutionary computation 6.2 (2002): 182-197.

The only changes in the algorithm is the replacement step that takes the dominance into consideration

## Multi-objective {.fragile}
\justifying

```{.haskell frame=lines framerule=2pt linenos=true fontsize=\footnotesize baselinestretch=0.8}
replacement pop =
  S = [filter (dominatedBy i) pop | i <- pop]
  n = [lengthIf (dominates i) pop | i <- pop]
  pop' = [i | i, ni <- enumerate n, ni == 0]
  case length pop'
    == length pop -> return pop'
    >  length pop -> crowding pop' (length pop)
    <  length pop -> return (pop' + replacement (pop / pop'))
```

## Multi-objective {.fragile}
\justifying

```{.haskell frame=lines framerule=2pt linenos=true fontsize=\footnotesize baselinestretch=0.8}
crowding pop nPop =
  dists = array 0 `ofSize` nPop
  for j <- [0 .. nObjs]
    ixs = argsortByObj j pop
    for (prev, i, next) <- window ixs
      dists[i] = dists[i] + (dists[next] - dists[prev])
    dists[first ix] = inf
    dists[last ixs] = inf
  ixs = argsort (-dists)
  return (take nPop ixs)
```

# Genetic Programming

## Evolving Programs
\justifying

Is it possible to evolve a computer program using just a sample of inputs and outputs?

## What is a program? {.fragile}
\justifying

A program can be thought as a function that gets one or more input arguments and return a value of a certain type.

## What is a program? {.fragile}
\justifying

This function can be decomposed in elementa functions that belong to our grammar:

```haskell
countElements [1, 2, 1, 3, 2, 1] = [(1,3), (2,2), (3,1)]

countElements = sort => group => map (head, length)
```

## What is a program? {.fragile}
\justifying

Depending of our context, the grammar can be reduced!

A classification problem has the following structure:

```haskell
if predicate
  then class1
  else class2
```

## What is a program? {.fragile}
\justifying

And the predicate is simply a boolean function:

```haskell
x1 > 3 && x2 <= 5
```

## What is a program? {.fragile}
\justifying

We only need to evolve the predicate function! We have `>, >=, <, <=, ==, !=, number, var, &&, ||, not` as part of our grammar.

## What is a program? {.fragile}
\justifying

A mathmatical expression is also a program:

```haskell
f(x) = x[1]^2 - x[2]*x[3]*cos(pi*x[1])
```

## Evolving a program
\justifying

To evolve a program we have first to define the set of functions and terminals:

| Name    | Set                                     |
|---------|-----------------------------------------|
|Function |$\left\{+,-,*,/,^2\right\}$              |
|Terminal |$\mathbb{R} \cup \left\{x1,x2,x3\right\}$|

## Evolving a program
\justifying

Each function requires a number of input arguments, this number is called **arity**.

The function $+$ has arity $2$, the partially applied function $^2$ has arity $1$.

The function `if-then-else` has arity $3$.

## Evolving a program
\justifying

These sets must obey the formal language rules:

- Every element of the terminal set $T$ must be a valid and correct expression.
- If $f \in F$ is a function with arity $n$ and $e_1, e_2, \ldots, e_n$ are  valid and correct expressions, then $f e_1 \ldots e_n$ is also a valid and correct expression.
- There are no other correct form besides these.

## Evolving a program
\justifying

It is also possible that the expression contains the type information.

A typed function must get the arguments of the correct type.

The expression $e_1 \lor e_2$ requires that $e_1, e_2$ are booleans.

## Genetic Programming
\justifying

The main evolutionary algorithm that evolves program is called **Genetic Programming** (GP).

## Genetic Programming {.fragile}
\justifying

```{.haskell frame=lines framerule=2pt linenos=true fontsize=\footnotesize baselinestretch=0.8}
gp =
  pop = randomPopulation
  until convergence do
    children = empty
    until length(children)==lambda do
      mut? = random(0,1)
      if mut?
        then ix       = random(0, n)
             child    = mutate(pop[ix])
             children = children <> child
        else (p1, p2) = randomParents(pop)
             (c1, c2) = combine(p1, p2)
             children = children <> [c1, c2]
    pop = replace pop children
return pop
```

## Evaluating program
\justifying

The fitness of a program is proportional to the amount of the test cases that passes within the training data.

If the output is a continuous value, we can measure the absolute or square difference between generated and expected outputs.

## Partial Functions
\justifying

If the representation allows invalid programs, there is a need to apply some treatments to either fix the program or protect the output.

For example, it is common to use the analytical quotient instead of division:

\begin{empheq}[box=\mybox]{align*}
aq(a, b) = \frac{a}{\sqrt{1 + b^2}}
\end{empheq}

## Representação
\justifying

There are many ways to represent a program:

- Linear
- Tree
- Direct Acyclic Graph

## Linear Representation
\justifying

With the linear representation, a program is reprenseted in its imperative form with state changes. In other words, a source code similar to an assembly language.

## Linear Representation
\justifying

This representation allows the programs to be coded as *bytecodes* allowing us to use the common mutation and recombination of binary representation.

## Linear Representation
\justifying

The Gene Expression Programming[^3] algorithm represents a program as an array of pre-fixed size:

\begin{empheq}[box=\mybox]{align*}
Q*-+abcd
\end{empheq}

[^3]: Ferreira, Candida. "Gene expression programming: a new adaptive algorithm for solving problems." arXiv preprint cs/0102027 (2001).

## Polish Notation
\justifying

We can also represent a function in polish notation:

\begin{empheq}[box=\mybox]{align*}
* + x 2 y = (x+2) * y
\end{empheq}

## Graph representation
\justifying

The algorithm *Cartesian Genetic Programming*[^4] represents a program as a graph:

![FONTE: http://www.cgplibrary.co.uk/files2/CartesianGeneticProgramming-txt.html](figs/cgp.png)

[^4]: Miller, Julian Francis, and Simon L. Harding. "Cartesian genetic programming." Proceedings of the 10th annual conference companion on Genetic and evolutionary computation. 2008.

## Tree Representation

![Fonte: Eiben, Agoston E., and James E. Smith. Introduction to evolutionary computing. Springer-Verlag Berlin Heidelberg, 2015.](figs/progTree.png){height=200px}

## Logic Tree

![Fonte: Eiben, Agoston E., and James E. Smith. Introduction to evolutionary computing. Springer-Verlag Berlin Heidelberg, 2015.](figs/logicTree.png){height=200px}

## Expression Tree

![Fonte: Eiben, Agoston E., and James E. Smith. Introduction to evolutionary computing. Springer-Verlag Berlin Heidelberg, 2015.](figs/exprTree.png){height=200px}

## Initialization of a Tree
\justifying

When creating an initial solution, we must be careful that we create a correct expression (i.e., the leaves are terminals) and that the program is not too large.

## Full method
\justifying

In the **full** method, we create a complete tree with depth $d$. In other words, all of the branches of the tree must have the same depth.

## Método Full {.fragile}

```{.haskell frame=lines framerule=2pt linenos=true fontsize=\footnotesize baselinestretch=0.8}
full max-depth =
  node = if max-depth == 0
           then sampleTerm
           else sampleNonTerm
  children = [full (max-depth - 1) | _ <- [1 .. arity(node)]]
  return (Node node children)
```

## Grow Method
\justifying

The **grow** method freely generates a tree up to a maximum depth $d$, where it will sample only terminals.

Before reaching the maximum depth, the sample of a node is biased toward non-terminals to avoid short programs.

## Método Grow {.fragile}

```{.haskell frame=lines framerule=2pt linenos=true fontsize=\footnotesize baselinestretch=0.8}
grow max-depth =
  ratio = n_terms / n_symbols
  r     = random(0,1)
  node  = if max-depth == 0 or r < ratio
            then sampleTerm
            else sampleNonTerm
  children = [grow (max-depth - 1) | _ <- [1 .. arity(node)]]
  return (Node node children)
```

## Ramped Half-and-Half

In **Ramped Half-and-Half** we create a population of initial solutions by varying the application of the previous methods with different values of maximum depths.

If we want to create $n$ individuals, we use full in half of them and grow to the other half. For each method, we choose a maximum depth from a range
$[min\_depth, max\_depth]$ uniformly distributed.

## Ramped Half-and-Half {.fragile}

```{.haskell frame=lines framerule=2pt linenos=true fontsize=\footnotesize baselinestretch=0.8}
ramped min-depth max-depth n-pop =
  range     = max-depth - min-depth + 1
  n         = n-pop / 2  -- divisão inteira
  (q, r)    = (n / 2, n % 2)
  treesFull = [full min-depth | _ <-[1..q]]
  treesGrow = [grow min-depth | _ <-[1..q+r]]
  trees     = ramped(min-depth+1, max-depth, n-pop - n)
  return (treesFull + treesGrow + trees)
```

# Mutation

## Subtree Replacement
\justifying

In this mutation operator we choose a subtree and replace it by a new randomly generated tree.

We can use either the grow or full method to create this random subtree.

## Subtree Replacement

\begin{columns}
\begin{column}{0.5\textwidth}
\begin{forest}
    rounded/.style={circle, draw},
    chosen/.style={circle, fill=gray, draw}
    [{$+$}, for tree=rounded
      [{$*$}
        [{$x_0$}]
        [{$2$}]
      ]
      [{$exp$}, for tree=chosen
        [,phantom]
        [{$\div$}
          [{$log$}
              [{$x_1$}]
              [,phantom]
          ]
          [{$3$}]
        ]
      ]
    ]
\end{forest}
\end{column}
\begin{column}{0.5\textwidth}
\begin{forest}
    rounded/.style={circle, draw},
    chosen/.style={circle, fill=gray, draw}
    [{$+$}, for tree=rounded
      [{$*$}
        [{$x_0$}]
        [{$2$}]
      ]
      [{$-$}, for tree=chosen
        [{$+$}
              [{$x_1$}]
              [{$x_0$}]
          [{$3$}]
        ]
        [{$2$}]
      ]
    ]
\end{forest}
\end{column}
\end{columns}

## Node Mutation
\justifying

With node mutation, we choose a node and change it with another token of the same arity.

## Node Mutation

\begin{columns}
\begin{column}{0.5\textwidth}
\begin{forest}
    rounded/.style={circle, draw},
    chosen/.style={circle, fill=gray, draw}
    [{$+$}, for tree=rounded
      [{$*$}
        [{$x_0$}]
        [{$2$}]
      ]
      [{$exp$}, chosen
        [,phantom]
        [{$\div$}
          [{$log$}
              [{$x_1$}]
              [,phantom]
          ]
          [{$3$}]
        ]
      ]
    ]
\end{forest}
\end{column}
\begin{column}{0.5\textwidth}
\begin{forest}
    rounded/.style={circle, draw},
    chosen/.style={circle, fill=gray, draw}
    [{$+$}, for tree=rounded
      [{$*$}
        [{$x_0$}]
        [{$2$}]
      ]
      [{$log$}, chosen
        [,phantom]
        [{$\div$}
          [{$log$}
              [{$x_1$}]
              [,phantom]
          ]
          [{$3$}]
        ]
      ]
    ]
\end{forest}
\end{column}
\end{columns}

## Swap Mutation
\justifying

In swap mutation, a non-terminal node is chosen and its children are swapped (if they are of the same type).

## Swap Mutation

\begin{columns}
\begin{column}{0.5\textwidth}
\begin{forest}
    rounded/.style={circle, draw},
    chosen/.style={circle, fill=gray, draw}
    [{$+$}, for tree=rounded
      [{$*$}, chosen
        [{$x_0$}]
        [{$2$}]
      ]
      [{$exp$}
        [,phantom]
        [{$\div$}
          [{$log$}
              [{$x_1$}]
              [,phantom]
          ]
          [{$3$}]
        ]
      ]
    ]
\end{forest}
\end{column}
\begin{column}{0.5\textwidth}
\begin{forest}
    rounded/.style={circle, draw},
    chosen/.style={circle, fill=gray, draw}
    [{$+$}, for tree=rounded
      [{$*$}
        [{$2$}, chosen ]
        [{$x_0$}, chosen ]
      ]
      [{$exp$}
        [,phantom]
        [{$\div$}
          [{$log$}
              [{$x_1$}]
              [,phantom]
          ]
          [{$3$}]
        ]
      ]
    ]
\end{forest}
\end{column}
\end{columns}

## Shrink Mutation
\justifying

In shrink mutation, a random nonterminal node is replaced by one of its children.

## Shrink Mutation

\begin{columns}
\begin{column}{0.5\textwidth}
\begin{forest}
    rounded/.style={circle, draw},
    chosen/.style={circle, fill=gray, draw}
    [{$+$}, for tree=rounded
      [{$*$}
        [{$x_0$}]
        [{$2$}]
      ]
      [{$*$}, chosen
        [{$*$}
          [{$x_0$}]
          [{$2$}]
        ]
        [{$\div$}
          [{$log$}
              [{$x_1$}]
              [,phantom]
          ]
          [{$3$}]
        ]
      ]
    ]
\end{forest}
\end{column}
\begin{column}{0.5\textwidth}
\begin{forest}
    rounded/.style={circle, draw},
    chosen/.style={circle, fill=gray, draw}
    [{$+$}, for tree=rounded
      [{$*$}
        [{$x_0$}]
        [{$2$}]
      ]
      [{$*$}, chosen
          [{$x_0$}, chosen ]
          [{$2$}, chosen ]
      ]
    ]
\end{forest}
\end{column}
\end{columns}

## Recombination
\justifying

The recombination operator simply chooses a subtree of each parent and swap them.

## Recombination

![Fonte: Eiben, Agoston E., and James E. Smith. Introduction to evolutionary computing. Springer-Verlag Berlin Heidelberg, 2015.](figs/cxGP.png)

# Bloat

## Bloat
\justifying

In the initial generations, it is common to observe the increase of the average fitness of the population together with the increase of the size of the tree.

At a certain point, it is possible to observe the increase in size without any increase in the average fitness.

## Bloat
\justifying

This phenomenon is known as **bloat** and it may be a problem since the computational cost to evaluate a larger program is higher.

Not only that, but the program loses its interpretability potential.

## Bloat
\justifying

A hypothesis for this is the **replication of accuracy theory** which says the ability to generate a child solution that is functionally similar to its parents favors its replication in the population.

Bloated expressions favors this property.

## Bloat
\justifying

In the **removal bias theory**, we notice that a tree can have inactive codes. When we apply reproduction and mutation in those inactive subtrees, there is a chance of increasing the size without any benefit to the fitness.

## Bloat
\justifying

The **programs search space theory** says that after a certain size, the average fitness will not change with size.

The reproduction of large programs tend to create even larger children which propagates throughout the generation favoring a population of large programs.

## Discarding bloat
\justifying

One way to deal with bloat is to disallow the generation of programs of a certain size.

If a children is larger than the maximum allowed size, it is discarded.

## Discarding bloat
\justifying

The main problem of this solution is that those trees close to the maximum allowed size will have many copies in the population as their children will likely be discarded.

## Discarding bloat
\justifying

Another solution is to return the children that violates such restrictions with a very small fitness such that it will be naturally discarded during replacement.

## Terminology learned today
\justifying

- **search problem:** find a valid solution from a set.
- **optimization problem:** find the best solution from a set.
- **maximization problem:** the best solution has the maximum value.
- **minimization problem:** the best solution has the minimum value.
- **search space:** set of all candidate solutions.
- **solution representation:** convenient representation of a solution.
- **objective-function:** functiona that maps a candidate to a value measuring the quality of the solution.
- **neighborhood:** set of candidate solutions close to a solution $s$.

## Terminology learned today
\justifying

- **local optima:** the best solution inside the neighborhood.
- **global optima:** the best solution of the search space.
- **local search:** searches for the nearest local optima.
- **heuristic:** technique to efficiently find a solution without any guarantees.
- **exploration:** act of exploring the search space.
- **exploitation:** act of exploring a promising neighborhood.

## Further reading
\justifying

- [Field Guide - GP](http://www0.cs.ucl.ac.uk/staff/W.Langdon/ftp/papers/poli08_fieldguide.pdf)
- [Livro - Koza](http://gpbib.cs.ucl.ac.uk/gp-html/koza_book.html)
- [Livro 2 - Koza](http://gpbib.cs.ucl.ac.uk/gp-html/koza_gp2.html)
- [Livro 3 - Koza](http://gpbib.cs.ucl.ac.uk/gp-html/koza_gp3.html)
- [Livro 4 - Koza](http://gpbib.cs.ucl.ac.uk/gp-html/koza_gp4.html)

## Further reading

- [Genetic Programming The Movie Part 1](https://youtu.be/tTMpKrKkYXo)
- [Genetic Programming The Movie Part 2](https://youtu.be/pRk6cth7Bpg)
- [Genetic Programming III: Human Competitive Machine Intelligence](https://youtu.be/8DY7akqFvfw)
- [Genetic Programming IV Video: Human-Competitive Machine Intelligence](https://youtu.be/R10GopQBsMc)
- Chapter 4 of Gabriel Kronberger, Bogdan Burlacu, Michael Kommenda, Stephan M.
Winkler, Michael Affenzeller. Symbolic Regression. To be plubished.

## Next lecture {.standout}

\begin{block}{}
  \begin{itemize}
    \item Nonlinear evolutionary symbolic regression
\end{itemize}

\hfill

\includegraphics[width=100pt]{figs/tobecontinued.png}
\end{block}

## Acknowledgments

