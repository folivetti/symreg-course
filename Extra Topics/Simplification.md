---
title: Model Simplification
author:
  - name: Prof. Fabrício Olivetti de França
institute: Federal University of ABC
logo:
  - name: figs/logotipo-ufabc-abaixo.eps
  - name: figs/hal.png
cover: figs/cover.png
date: 15 June, 2024
---

# Simplifying Algebraic Expressions

## Simplifying Algebraic Expressions

\notebox{Rewriting an algebraic expression by applying equivalence
rules to generate an equivalent expression that is more efficient and compact. }

## Equivalent Expressions

\notebox{\textbf{Equivalent expressions} are those that are equal in value, even if they look different.}

## Equivalent Expressions

\begin{empheq}[box=\mybox]{align*}
x + y &\equiv y + x \\
2 x &\equiv x + x \\
\tanh x &\equiv \frac{e^{2x} - 1}{e^{2x} + 1}
\end{empheq}

## Efficiency and Compactness
\justifying

There is no standard measure of efficiency and compactness for an algebraic expression
as this depends on the application:

- Smaller expressions
- Computationally more efficient
- Less repeated use of the same variable
- Less numerical constants
- etc.

## Efficiency and Compactness
\justifying

For example:

- $x+x+x$ could be much faster to calculate than $3x$ in older architectures.
- $3x$ is more precise if we are using interval arithmetic.
- $(x + 1.5y)/(1 + 3x)$ can be more interpretable than $(2x + 3y)/(2 + 6x)$ (if these parameters can still fit the data).

## Equivalence Rules
\justifying

This simplification process depends on a set of equivalence rules that describes whether two expressions are equivalent.
Let's see a couple more examples:

\begin{empheq}[box=\mybox]{align*}
x + y &\equiv y + x \\
xy &\equiv yx \\
2 x &\equiv x + x \\
x + (y + z) &\equiv (x + y) + z \\
x (y + z) &\equiv xy + xz \\
(xy)/z &\equiv x(y/z)
\end{empheq}

## Optimal Expression
\justifying

Finding the optimal expression is not simply applying these rules sequentially.
The order of the application matters, enumerating all possible orders leads to
a combinatorial explosion.

If we have $N$ rules and we can only apply them once, we have $N!$ ways of applying these rules.

Of course, not every rule is appliable at every moment and we may need to apply a rule more than once.

## Program Optimization
\justifying

This is a well known problem in compiler optimization called **Phase Ordering Problem**:

\notebox{Given a measure of goodness for each rule at a given expression, if two rules give the same local
benefit, then both can lead to sub-optimal results.}

## Program Optimization
\justifying

Let's see this problem using algebraic expressions $(2x)/2$ and the following rules:

\begin{empheq}[box=\mybox]{align*}
2 \alpha &\equiv \alpha << 1\\
\alpha \beta &\equiv \beta\alpha \\
1 \alpha &\equiv \alpha \\
\alpha / \alpha &\equiv 1\\
(\alpha\beta)/\gamma &\equiv \alpha(\beta/\gamma)
\end{empheq}

## Program Optimization
\justifying

:::::::::::::: {.columns}
::: {.column width="50%"}
\begin{empheq}[box=\mybox]{align*}
\hlmath{yellow}{eq1}{2 \alpha &\equiv \alpha << 1}\\
\alpha \beta &\equiv \beta\alpha \\
1 \alpha &\equiv \alpha \\
\alpha / \alpha &\equiv 1\\
(\alpha\beta)/\gamma &\equiv \alpha(\beta/\gamma)
\end{empheq}
:::
::: {.column width="50%"}
\begin{empheq}[box=\mybox]{align*}
\alpha &= x
\end{empheq}

\begin{empheq}[box=\mybox]{align*}
(2x)/2 &= (x << 1)/2
\end{empheq}
:::
::::::::::::::


## Program Optimization
\justifying

:::::::::::::: {.columns}
::: {.column width="50%"}
\begin{empheq}[box=\mybox]{align*}
2 \alpha &\equiv \alpha << 1\\
\hlmath{yellow}{eq2}{\alpha \beta &\equiv \beta\alpha} \\
1 \alpha &\equiv \alpha \\
\alpha / \alpha &\equiv 1\\
(\alpha\beta)/\gamma &\equiv \alpha(\beta/\gamma)
\end{empheq}
:::
::: {.column width="50%"}
\begin{empheq}[box=\mybox]{align*}
\end{empheq}

\begin{empheq}[box=\mybox]{align*}
(2x)/2 &= (x << 1)/2
\end{empheq}
:::
::::::::::::::


## Program Optimization
\justifying

:::::::::::::: {.columns}
::: {.column width="50%"}
\begin{empheq}[box=\mybox]{align*}
2 \alpha &\equiv \alpha << 1\\
\alpha \beta &\equiv \beta\alpha \\
\hlmath{yellow}{eq3}{1 \alpha &\equiv \alpha} \\
\alpha / \alpha &\equiv 1\\
(\alpha\beta)/\gamma &\equiv \alpha(\beta/\gamma)
\end{empheq}
:::
::: {.column width="50%"}
\begin{empheq}[box=\mybox]{align*}
\end{empheq}

\begin{empheq}[box=\mybox]{align*}
(2x)/2 &= (x << 1)/2
\end{empheq}
:::
::::::::::::::


## Program Optimization
\justifying

:::::::::::::: {.columns}
::: {.column width="50%"}
\begin{empheq}[box=\mybox]{align*}
2 \alpha &\equiv \alpha << 1\\
\alpha \beta &\equiv \beta\alpha \\
1 \alpha &\equiv \alpha \\
\hlmath{yellow}{eq4}{\alpha / \alpha &\equiv 1}\\
(\alpha\beta)/\gamma &\equiv \alpha(\beta/\gamma)
\end{empheq}
:::
::: {.column width="50%"}
\begin{empheq}[box=\mybox]{align*}
\end{empheq}

\begin{empheq}[box=\mybox]{align*}
(2x)/2 &= (x << 1)/2
\end{empheq}
:::
::::::::::::::


## Program Optimization
\justifying

:::::::::::::: {.columns}
::: {.column width="50%"}
\begin{empheq}[box=\mybox]{align*}
2 \alpha &\equiv \alpha << 1\\
\alpha \beta &\equiv \beta\alpha \\
1 \alpha &\equiv \alpha \\
\alpha / \alpha &\equiv 1\\
\hlmath{yellow}{eq5}{\left(\alpha\beta\right)/\gamma &\equiv \alpha\left(\beta/\gamma\right)}
\end{empheq}
:::
::: {.column width="50%"}
\begin{empheq}[box=\mybox]{align*}
\end{empheq}

\begin{empheq}[box=\mybox]{align*}
(2x)/2 &= (x << 1)/2
\end{empheq}
:::
::::::::::::::


## Program Optimization
\justifying

That was quick and sub-optimal! Let's try again:

:::::::::::::: {.columns}
::: {.column width="50%"}
\begin{empheq}[box=\mybox]{align*}
\hlmath{yellow}{eq6}{\alpha \beta &\equiv \beta\alpha} \\
(\alpha\beta)/\gamma &\equiv \alpha (\beta / \gamma) \\
\alpha / \alpha &\equiv 1\\
1 \alpha &\equiv \alpha \\
2 \alpha &\equiv \alpha << 1
\end{empheq}
:::
::: {.column width="50%"}
\begin{empheq}[box=\mybox]{align*}
\alpha &= 2 \\
\beta &= x
\end{empheq}

\begin{empheq}[box=\mybox]{align*}
(2x)/2 &= (x 2) / 2
\end{empheq}
:::
::::::::::::::


## Program Optimization
\justifying

:::::::::::::: {.columns}
::: {.column width="50%"}
\begin{empheq}[box=\mybox]{align*}
\alpha \beta &\equiv \beta\alpha \\
\hlmath{yellow}{eq7}{\left(\alpha\beta\right)/\gamma &\equiv \alpha\left(\beta/\gamma\right)} \\
\alpha / \alpha &\equiv 1\\
1 \alpha &\equiv \alpha \\
2 \alpha &\equiv \alpha << 1
\end{empheq}
:::
::: {.column width="50%"}
\begin{empheq}[box=\mybox]{align*}
\alpha &= x \\
\beta &= 2 \\
\gamma &= 2
\end{empheq}

\begin{empheq}[box=\mybox]{align*}
(2x)/2 &= (x 2) / 2 \\
(x 2) / 2 &= x (2 / 2)
\end{empheq}
:::
::::::::::::::



## Program Optimization
\justifying

:::::::::::::: {.columns}
::: {.column width="50%"}
\begin{empheq}[box=\mybox]{align*}
\alpha \beta &\equiv \beta\alpha \\
(\alpha\beta)/\gamma &\equiv \alpha(\beta/\gamma) \\
\hlmath{yellow}{eq8}{\alpha / \alpha &\equiv 1}\\
1 \alpha &\equiv \alpha \\
2 \alpha &\equiv \alpha << 1
\end{empheq}
:::
::: {.column width="50%"}
\begin{empheq}[box=\mybox]{align*}
\alpha &= 2
\end{empheq}

\begin{empheq}[box=\mybox]{align*}
(2x)/2 &= (x 2) / 2 \\
(x 2) / 2 &= x (2 / 2) \\
x (2 / 2) &= x 1
\end{empheq}
:::
::::::::::::::


## Program Optimization
\justifying

:::::::::::::: {.columns}
::: {.column width="50%"}
\begin{empheq}[box=\mybox]{align*}
\alpha \beta &\equiv \beta\alpha \\
(\alpha\beta)/\gamma &\equiv \alpha(\beta/\gamma) \\
\alpha / \alpha &\equiv 1\\
\hlmath{yellow}{eq9}{1 \alpha &\equiv \alpha} \\
2 \alpha &\equiv \alpha << 1
\end{empheq}
:::
::: {.column width="50%"}
\begin{empheq}[box=\mybox]{align*}
\end{empheq}

\begin{empheq}[box=\mybox]{align*}
(2x)/2 &= (x 2) / 2 \\
(x 2) / 2 &= x (2 / 2) \\
x (2 / 2) &= x 1
\end{empheq}
:::
::::::::::::::

## Program Optimization
\justifying

So close! But if we go through the rules again we will reach the optimal result!

:::::::::::::: {.columns}
::: {.column width="50%"}
\begin{empheq}[box=\mybox]{align*}
\alpha \beta &\equiv \beta\alpha \\
(\alpha\beta)/\gamma &\equiv \alpha(\beta/\gamma) \\
\alpha / \alpha &\equiv 1\\
1 \alpha &\equiv \alpha \\
\hlmath{yellow}{eq13}{2 \alpha &\equiv \alpha << 1}
\end{empheq}
:::
::: {.column width="50%"}
\begin{empheq}[box=\mybox]{align*}
\end{empheq}

\begin{empheq}[box=\mybox]{align*}
(2x)/2 &= (x 2) / 2 \\
(x 2) / 2 &= x (2 / 2) \\
x (2 / 2) &= x 1
\end{empheq}
:::
::::::::::::::

## Program Optimization
\justifying

:::::::::::::: {.columns}
::: {.column width="50%"}
\begin{empheq}[box=\mybox]{align*}
\hlmath{yellow}{eq10}{\alpha \beta &\equiv \beta\alpha} \\
(\alpha\beta)/\gamma &\equiv \alpha(\beta/\gamma) \\
\alpha / \alpha &\equiv 1\\
1 \alpha &\equiv \alpha \\
2 \alpha &\equiv \alpha << 1
\end{empheq}
:::
::: {.column width="50%"}
\begin{empheq}[box=\mybox]{align*}
\alpha &= x \\
\beta &= 1
\end{empheq}

\begin{empheq}[box=\mybox]{align*}
(2x)/2 &= (x 2) / 2 \\
(x 2) / 2 &= x (2 / 2) \\
x (2 / 2) &= x 1 \\
x 1 &= 1 x
\end{empheq}
:::
::::::::::::::

## Program Optimization
\justifying

Good!

:::::::::::::: {.columns}
::: {.column width="50%"}
\begin{empheq}[box=\mybox]{align*}
\alpha \beta &\equiv \beta\alpha \\
(\alpha\beta)/\gamma &\equiv \alpha(\beta/\gamma) \\
\alpha / \alpha &\equiv 1\\
\hlmath{yellow}{eq11}{1 \alpha &\equiv \alpha} \\
2 \alpha &\equiv \alpha << 1
\end{empheq}
:::
::: {.column width="50%"}
\begin{empheq}[box=\mybox]{align*}
\alpha &= x
\end{empheq}

\begin{empheq}[box=\mybox]{align*}
(2x)/2 &= (x 2) / 2 \\
(x 2) / 2 &= x (2 / 2) \\
x (2 / 2) &= x 1 \\
x 1 &= 1 x \\
1 x &= x
\end{empheq}
:::
::::::::::::::

## Destroy the Past!
\justifying

One reason for the phase ordering problem is that the program transformation
is destructive.

Whenever we apply a rule, we *forget* about the past equivalent expressions.
So, whenever we take the wrong *path*, we cannot go back.

# Equality Saturation

## Non-destructive rewriting
\justifying

In [^1] Tate introduced the idea of **Equality Saturation** proposing a method for non-destructive rewriting of the original expression.

The main idea is that all the rules are applied in parallel and the resulting expressions are kept in a compact data structure
now knownas **e-graph**.

[^1]: Tate, R., et al, "Equality Saturation: a New Approach to Optimization," in Logical Methods in Computer Science, 2011.

## Non-destructive rewriting
\justifying

This technique became popular after a fast and customizable implementation called *egg*[^2].
Egg is implemented in Rust and enabled the user to experiment with equality saturation in different applications with
the requirements of:

- Defining a Language
- Writing the rules

[^2]: Willsey, M., et al. "egg: Fast and extensible equality saturation," in Proceedings of the ACM on Programming Languages, vol. 5, no. POPL, pp. 1–29, 2021

## Equality Saturation
\justifying

An e-graph[^3] is composed of:

- a set of e-classes (dashed) with a non-empty sets of e-nodes (solid line).
- a set of e-nodes that represents one symbol of our language
- a set of edges connecting one e-node to one or more e-classes.

## Equality Saturation
\justifying

\begin{figure}[th!]
\centering
\begin{subfigure}[b]{0.25\textwidth}
%\centering
\begin{tikzpicture}
    \node[shape=rectangle,draw=black] (plus) {$+$};
    \node[shape=rectangle,draw=black, below=of plus.west] (times) {$*$};
    \node[shape=rectangle,draw=black, below=of times.west] (one) {$1$};
    \node[shape=rectangle,draw=black, right=of one.east] (x) {$x$};
    \node[shape=rectangle,dashed,draw=black,fit={(plus)}] (nplus) {};
    \node[shape=rectangle,dashed,draw=black,fit={(times)}] (ntimes) {};
    \node[shape=rectangle,dashed,draw=black,fit={(one)}] (none) {};
    \node[shape=rectangle,dashed,draw=black,fit={(x)}] (nx) {};

    \path [->] (plus) edge (ntimes);
    \path [->] (times) edge (none);
    \path [->] (times) edge (nx);
    \path [->] (plus) edge (nx);
\end{tikzpicture}
\caption{}
\label{fig:egraph-a}
\end{subfigure}
\hfill
\begin{subfigure}[b]{0.25\textwidth}
%\centering
\begin{tikzpicture}
    \node[shape=rectangle,draw=black] (plus) {$+$};
    \node[shape=rectangle,draw=black, below=of plus.west] (times) {$*$};
    \node[shape=rectangle,draw=black, right=of times.west] (x1) {$x$};
    \node[shape=rectangle,draw=black, below=of times.west] (one) {$1$};
   % \node[shape=rectangle,draw=black, right=of one.east] (x2) {$x$};
    \node[shape=rectangle,dashed,draw=black,fit={(plus)}] (nplus) {};
    \node[shape=rectangle,dashed,draw=black,fit={(times) (x1)}] (ntimes) {};
    \node[shape=rectangle,dashed,draw=black,fit={(one)}] (none) {};
   % \node[shape=rectangle,dashed,draw=black,fit={(x2)}] (nx) {};

    \path [->] (plus) edge[bend left= 20] (ntimes);
    \path [->] (times) edge (none);
    \path [->] (times) edge[in=90,out=-160,loop] (ntimes);
    \path [->] (plus) edge[bend right= 20] (ntimes);
\end{tikzpicture}
\caption{}
\label{fig:egraph-b}
\end{subfigure}
\hfill
\begin{subfigure}[b]{0.25\textwidth}
%\centering
\begin{tikzpicture}
    \node[shape=rectangle,draw=black] (plus) {$+$};
    \node[shape=rectangle,draw=black, right=of plus.west] (times1) {$*$};
    \node[shape=rectangle,draw=black, below=of plus.west] (times2) {$*$};
    \node[shape=rectangle,draw=black, right=of times2.west] (x1) {$x$};
    \node[shape=rectangle,draw=black, below=of times2.west] (one) {$1$};
   % \node[shape=rectangle,draw=black, right=of one.east] (x2) {$x$};
    \node[shape=rectangle,dashed,draw=black,fit={(plus) (times1)}] (nplus) {};
    \node[shape=rectangle,dashed,draw=black,fit={(times) (x1)}] (ntimes) {};
    \node[shape=rectangle,dashed,draw=black,fit={(one)}] (none) {};
   % \node[shape=rectangle,dashed,draw=black,fit={(x2)}] (nx) {};
    \node[shape=rectangle,draw=black, right=of one.west] (two) {$2$};
    \node[shape=rectangle,dashed,draw=black,fit={(two)}] (ntwo) {};

    \path [->] (plus) edge[bend left= 20] (ntimes);
    \path [->] (times2) edge (none);
    \path [->] (times2) edge[in=90,out=-160,loop] (ntimes);
    \path [->] (plus) edge[bend right= 20] (ntimes);
    \path [->] (times1) edge[bend left=30] (ntwo);
    \path [->] (times1) edge (ntimes);
\end{tikzpicture}

\caption{}
\label{fig:egraph-c}
\end{subfigure}
\caption{(a) Original expression; (b) after applying the rule $1 * x \rightarrow x$; (c) after applying the rule $x + x \rightarrow 2 * x$.}
\label{fig:egraph}
\end{figure}

[^3]: Nelson, Charles Gregory. Techniques for program verification. Stanford University, 1980.

## Equality Saturation
\justifying

Notice how the children of an e-node points to the e-classes that contain
one or more equivalent sub-expression.
For example, we can go from $+$ to $1x$ or $x$ or $1 \times 1 \times x$.


\begin{tikzpicture}
    \node[shape=rectangle,draw=black] (plus) {$+$};
    \node[shape=rectangle,draw=black, right=of plus.west] (times1) {$*$};
    \node[shape=rectangle,draw=black, below=of plus.west] (times2) {$*$};
    \node[shape=rectangle,draw=black, right=of times2.west] (x1) {$x$};
    \node[shape=rectangle,draw=black, below=of times2.west] (one) {$1$};
   % \node[shape=rectangle,draw=black, right=of one.east] (x2) {$x$};
    \node[shape=rectangle,dashed,draw=black,fit={(plus) (times1)}] (nplus) {};
    \node[shape=rectangle,dashed,draw=black,fit={(times) (x1)}] (ntimes) {};
    \node[shape=rectangle,dashed,draw=black,fit={(one)}] (none) {};
   % \node[shape=rectangle,dashed,draw=black,fit={(x2)}] (nx) {};
    \node[shape=rectangle,draw=black, right=of one.west] (two) {$2$};
    \node[shape=rectangle,dashed,draw=black,fit={(two)}] (ntwo) {};

    \path [->] (plus) edge[bend left= 20] (ntimes);
    \path [->] (times2) edge (none);
    \path [->] (times2) edge[in=90,out=-160,loop] (ntimes);
    \path [->] (plus) edge[bend right= 20] (ntimes);
    \path [->] (times1) edge[bend left=30] (ntwo);
    \path [->] (times1) edge (ntimes);
\end{tikzpicture}

## Equality Saturation
\justifying

At a certain point, whenever we apply the rules the graph will not change!
This means we reached the fixed point and the graph is saturated.

The saturated graph represents every equivalent expression reachable by
the specific set of rules.


\begin{tikzpicture}
    \node[shape=rectangle,draw=black] (plus) {$+$};
    \node[shape=rectangle,draw=black, right=of plus.west] (times1) {$*$};
    \node[shape=rectangle,draw=black, below=of plus.west] (times2) {$*$};
    \node[shape=rectangle,draw=black, right=of times2.west] (x1) {$x$};
    \node[shape=rectangle,draw=black, below=of times2.west] (one) {$1$};
   % \node[shape=rectangle,draw=black, right=of one.east] (x2) {$x$};
    \node[shape=rectangle,dashed,draw=black,fit={(plus) (times1)}] (nplus) {};
    \node[shape=rectangle,dashed,draw=black,fit={(times) (x1)}] (ntimes) {};
    \node[shape=rectangle,dashed,draw=black,fit={(one)}] (none) {};
   % \node[shape=rectangle,dashed,draw=black,fit={(x2)}] (nx) {};
    \node[shape=rectangle,draw=black, right=of one.west] (two) {$2$};
    \node[shape=rectangle,dashed,draw=black,fit={(two)}] (ntwo) {};

    \path [->] (plus) edge[bend left= 20] (ntimes);
    \path [->] (times2) edge (none);
    \path [->] (times2) edge[in=90,out=-160,loop] (ntimes);
    \path [->] (plus) edge[bend right= 20] (ntimes);
    \path [->] (times1) edge[bend left=30] (ntwo);
    \path [->] (times1) edge (ntimes);
\end{tikzpicture}

# egg: Fast and Extensible Equality Saturation

## egg in a Nutshell {.fragile}

```{.python frame=lines framerule=2pt linenos=true fontsize=\footnotesize baselinestretch=0.8}
def equality_saturation(expr, rewrites):
    egraph = build_egraph(expr)
    while !egraph.is_saturated_or_timeout():
        matches = []

        # read-only
        for rw in rewrites:
            for (subst, eclass) in egraph.search(rw.lhs):
                matches.append((rw, subst, eclass))

        # write-only
        for (rw, subst, eclass) in matches:
            new_eclass = egraph.add(rw.rhs.subst(subst))
            egraph.union(eclass, new_eclass)

        # restore invariants
        egraph.rebuild()

    return egraph.extract_best()
```

## Equality Saturation
\justifying

In the following, let's use as an illustrative example
the expression $(2 / x) (x + x)$ and the rules:

\begin{empheq}[box=\mybox]{align*}
x + x &\rightarrow 2x \\
(x/y)z &\rightarrow x(z/y) \\
(xy)/z &\rightarrow x(y/z)\\
x/x &\rightarrow 1\\
x \cdot 1 &\rightarrow x
\end{empheq}

## e-node representation
\justifying

The representation of an e-node follows the fixed point of the representation of the program or expression.

In mathematics, a **fixed point** $x$ of a function $f$, also known as **invariant point** is a value $x$ such that $f(x) = x$. The **least fixed point** is the smallest fixed point $x$ among all fixed points of a function.

## e-node representation {.fragile}
\justifying

As a data structure we can represent a fixed point as:

```{.haskell frame=lines framerule=2pt linenos=true fontsize=\footnotesize baselinestretch=0.8}
data Fix f = Fix (f (Fix f))
```

which allows us to implement generic algorithms for folding and unfolding data structures.

## e-node representation {.fragile}
\justifying

An explicit fixed point definition for an expression tree is:

```{.haskell frame=lines framerule=2pt linenos=true fontsize=\footnotesize baselinestretch=0.8}
data Expr a = Const Double
            | Var String
            | Add a a
            | Sub a a
            | Mul a a
            | Div a a

type FixExpr = Fix Expr
```

Notice that this structure allows a recursive pattern by replacing the type parameter `a` with `FixExpr`. Also, we may naturally represent an `Add` node that points to an e-class id by making `Add 1 2 :: Expr Int`.

## e-node representation {.fragile}
\justifying

In dynamically typed language this can be expressed as structures with branches that can assume different
types as it seems fit.

```{.python frame=lines framerule=2pt linenos=true fontsize=\footnotesize baselinestretch=0.8}
class Add:
    left: Any
    right: Any

class Mul:
    left: Any
    right: Any

class Var:
    name: String

class Const:
    val: float

Expr = Add | Mul | Var | Const
```

## e-node representation
\justifying

This definition allows us to represent our expression as an expression tree
where each children of a node is another tree or an integer representing the e-class id it points to
or a bottom symbol.

We will see next how this flexibility is useful for the implementation of egg.

## Expression to e-node
\justifying

Starting with our exression in a tree-like representation:

\centering
![](figs/egraph-expr_to_egraph_1.drawio.png){width=150px}

## Expression to e-node
\justifying

Visiting the nodes in a post-order traversal...

\centering
![](figs/egraph-expr_to_egraph_2.drawio.png){width=150px}

## Expression to e-node
\justifying

We create an e-node storing it into a new e-class and assign an e-class id.

\centering
![](figs/egraph-expr_to_egraph_3.drawio.png){width=250px}

## Expression to e-node
\justifying

We create an e-node storing it into a new e-class and assign an e-class id.

\centering
![](figs/egraph-expr_to_egraph_4.drawio.png){width=250px}

## Expression to e-node
\justifying

Any internal node will have their children replace by the corresponding e-class ids.
We keep a hash table of this node representation to the corresponding assigned id.

\centering
![](figs/egraph-expr_to_egraph_5.drawio.png){width=250px}

## Expression to e-node
\justifying

Whenever we reach a node contained in the hash table, we just retrieve the e-class id it belongs to.

\centering
![](figs/egraph-expr_to_egraph_6.drawio.png){width=250px}

## Expression to e-node
\justifying

This will ensure that the graph is compact and the equivalent nodes are grouped together.

\centering
![](figs/egraph-expr_to_egraph_7.drawio.png){width=250px}

## Expression to e-node
\justifying

With this representation, it becomes easier to search the patterns for a match.

\centering
![](figs/egraph-expr_to_egraph_8.drawio.png){width=250px}

## Equality Saturation
\justifying

The corresponding e-graph of the expression $(2 / x) (x + x)$ is

\centering
![](figs/egraph_1.png){width=150px}

## Equality Saturation
\justifying

By applying the rule $x + x \rightarrow 2x$ we get

\centering
![](figs/egraph_2.png){width=150px}

## Equality Saturation
\justifying

By applying the rule $(x/y)z \rightarrow x(z/y)$ we get

\centering
![](figs/egraph_3.png){width=250px}

## Equality Saturation
\justifying

After this iteration, we must merge the e-nodes that are equivalent to the same e-class:

\centering
![](figs/egraph_4.png){width=150px}

Notice how following any path from a given e-class will return equivalent expressions.

## Equality Saturation
\justifying

After applying every rule, we start again ignoring the already performed rewritings.
Now we can match $(xy)/z \rightarrow x(y/z)$:

\centering
![](figs/egraph_5.png){width=250px}

## Equality Saturation
\justifying

And now $x/x \rightarrow 1$:

\centering
![](figs/egraph_6.png){width=250px}

## Equality Saturation
\justifying

Finally, $x \cdot 1 \rightarrow x$:

\centering
![](figs/egraph_7.png){width=250px}

## Matching rules
\justifying

The pattern matching procedure starts with the creation of a database of patterns.
This database is composed of a hash table where the keys are each node with their children
replaced by a unit value, and the values are a sequence of Tries linking to the e-class of id of each
element.

\centering
![](figs/egraph-db1.drawio.png){width=250px}

## Matching rules
\justifying

For example, the entry for `*` contains two e-classes: `1, 3`. The first child of e-class `1` can
be either `2` or `5` depending of what e-node is chosen. This follows up until for every child of those
e-nodes.

\centering
![](figs/egraph-db1.drawio.png){width=250px}

## Matching rules
\justifying

If we want to match the rule `("x" / "y") * "z"`, we first retrieve the patterns `_ * _` from the hash table, finding that it is found at e-classes `1` and `3`.

\centering
![](figs/egraph-match.drawio.png){width=250px}

## Matching rules
\justifying

Next, we retrieve the left child of these e-classes, generating the set `{2, 5, 6}`.

\centering
![](figs/egraph-match.drawio.png){width=250px}

## Matching rules
\justifying

Finally, we retrieve the entry `_ / _` from the hash table, finding the e-classes `{2,4}`.
Intersecting both sets $\left\{2,5,6\right\} \cap \left\{2, 4\right\} = \left\{2\right\}$ we find
the single match.

\centering
![](figs/egraph-match.drawio.png){width=250px}


## Extracting the Optimal Expression
\justifying

After reaching saturation, we can extract the optimal expressions
by following a greedy heuristic or any graph traversal algorithm
we find suitable.

\centering
![](figs/egraph-getbest.drawio.png){width=250px}

## Extracting the Optimal Expression
\justifying

For example, if we say that $+$ has cost $2$ and $*, /$ have cost $3$, and
terminals have cost $1$. By greedly following the post-order traversal, we would
assign a partial result for each e-class as the minimum result from all e-nodes.

\centering
![](figs/egraph-getbest.drawio.png){width=250px}

## Extracting the Optimal Expression
\justifying

In our example, we would find $2 * 2$ as the best expression.

\centering
![](figs/egraph-getbest.drawio.png){width=250px}


## Constant Folding
\justifying

Other interesting features from *egg* is the constant folding, which allows us to
store facts about each e-class. In our example, we can store the constant that this
particular e-class can be evaluated to, if any.

\centering
![](figs/egraph_8.png){width=250px}

## Constant Folding
\justifying

In this situation, we could simply add the e-node $4$ into the root e-class, finding an
even smaller expression.

\centering
![](figs/egraph_8.png){width=250px}

## Additional resources (clickable links)
\justifying

- [egraphs-good website](https://egraphs-good.github.io/)
- [egg](https://github.com/egraphs-good/egg)
- [hegg (Haskell)](https://github.com/alt-romes/hegg)
- [Metatheory.jl (Julia)](https://github.com/JuliaSymbolics/Metatheory.jl)
- [egg presentation](https://www.youtube.com/watch?v=6cJMI9z2TeU)

# Application in Symbolic Regression


## Overparameterized models

Symbolic Regression algorithms are prone to overparametrization[^4]:

$$
f(\mathbf{x, \theta}) = \theta_1 \exp{(\theta_2 x_1 + \theta_3)}
$$

- The parameters can assume different values for the same data
- Numerical issues and slow convergence of optimization
- Larger search space for memetic approaches
- Interpretation is hindered

[^4]: de Franca, Fabricio Olivetti, and Gabriel Kronberger. "Reducing Overparameterization of Symbolic Regression Models with Equality Saturation." Proceedings of the Genetic and Evolutionary Computation Conference. 2023.

## Automatic Simplification

Consider the expression

$$
\frac{\theta_1(\theta_2 x + \theta_3 y)}{\theta_4}
$$

And the simplification rules:

\begin{align}
\text{distributivity: } & a (b + c) = ab + ac \\
\text{commutativity: } & a b = b a \\
\text{associativity: } & (ab)/c = a(b/c)
\end{align}

## Automatic Simplification

Applying the rules in order will lead to:

$$
\frac{\theta_1(\theta_2 x + \theta_3 y)}{\theta_4}
$$

$$
\frac{\theta_1\theta_2 x + \theta_1 \theta_3 y}{\theta_4}
$$

$$
\frac{\gamma_1 x + \gamma_2 y}{\gamma_3}
$$

## Automatic Simplification

Applying rules $2, 3, 2, 1$ we get:

$$
\frac{\theta_1(\theta_2 x + \theta_3 y)}{\theta_4}
$$

$$
\frac{(\theta_2 x + \theta_3 y)\theta_1}{\theta_4}
$$

$$
(\theta_2 x + \theta_3 y)\gamma_1
$$

$$
\gamma_1(\theta_2 x + \theta_3 y)
$$

$$
\gamma_1x + \gamma_2y
$$



## Research Questions

- Is overparametrization really a problem?
- Can EqSat help to reduce overparametrization in SR?
  - How well does it compare with simpler alternatives? (e.g., Sympy)
- Does EqSat always reduce to the optimal number of parameters?
- How fast is EqSat?

## Is this a real problem?

- Run MOO version of Operon once for every Feynman dataset
- Stored the Pareto front with a total of $183\,491$ models

\begin{figure}[t!]
    \centering
\includegraphics[trim={0.1cm 1.0cm 0.1cm 0.1cm},clip,width=0.55\textwidth]{figs/operon_eqsat.pdf}
\end{figure}

## How much can EqSat help?

- 30 independent runs of a set of SR algorithms to:

\begin{align}
    f_1(x, y) &= \frac{1}{1 + x^{-4}} + \frac{1}{1 + y^{-4}} \\
    f_2(x, y) &= \frac{e^{-(x - 1)^2}}{1.2 + (y - 2.5)^2}
\end{align}

:::columns
::::column
- Simplify with:
  - EqSat
  - Sympy
  - Sympy + EqSat
::::
::::column
- SR algorithms:
  - Bingo
  - EPLEX
  - GP-GOMEA
  - Operon
  - PySR
  - SBP
::::
:::

## How much can EqSat help?

\begin{figure}[th!]
\centering
\begin{subfigure}[b]{0.9\textwidth}
\includegraphics[trim={0.1cm 0.0cm 0.1cm 0.1cm},clip]{figs/boxplot_pagie_ratio.pdf}
\end{subfigure}
\begin{subfigure}[b]{0.9\textwidth}
\includegraphics[trim={0.1cm 0.0cm 0.1cm 0.1cm},clip]{figs/boxplot_kotanchek_ratio.pdf}
\end{subfigure}
\caption{Ratio of decrease}
\end{figure}

## Is it optimal?

\% simplified expressions with \# parameters equal to rank (left) or with at most one extra (right).

:::columns
::::column
| \textbf{Algorithm} | \textbf{Pagie-1} | \textbf{Kotanchek} |
|:------------------:|:----------------:|:------------------:|
|        Bingo       |     $27\%$    |      $22\%$     |
|        EPLEX       |     $28\%$    |      $\mathbf{18\%}$     |
|      GP-GOMEA      |     $30\%$    |      $\mathbf{76\%}$     |
|       Operon       |     $66\%$    |      $74\%$     |
|        PySR        |     $36\%$    |      $34\%$     |
|         SBP        |     $42\%$    |      $60\%$     |
::::
::::column
| \textbf{Algorithm} | \textbf{Pagie-1} | \textbf{Kotanchek} |
|:------------------:|:----------------:|:------------------:|
|        Bingo       |     $\mathbf{33\%}$    |      $66\%$     |
|        EPLEX       |     $45\%$    |      $37\%$     |
|      GP-GOMEA      |    $\mathbf{100\%}$    |     $\mathbf{100\%}$     |
|       Operon       |    $\mathbf{100\%}$    |      $94\%$     |
|        PySR        |     $52\%$    |      $71\%$     |
|         SBP        |     $60\%$    |     $\mathbf{100\%}$     |
::::
:::


## Is it fast?

\begin{figure}[t!]
    \centering
\includegraphics[trim={0.1cm 0.0cm 0.1cm 0.1cm},clip,width=0.7\textwidth]{figs/runtime.pdf}
\end{figure}

## Discussions

- Overparametrization is widespread in SymReg
- EqSat can consistently reduce the number of parameters
- Sympy sometimes makes it worse
- Runtime is low most of the time and it can be improved

## Next lecture {.standout}

\begin{block}{}
  \begin{itemize}
    \item Confidence Intervals with Profile Likelihood
\end{itemize}

\hfill

\includegraphics[width=100pt]{figs/tobecontinued.png}
\end{block}

## Acknowledgments
