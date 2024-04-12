---
title: Model Symplification
author:
  - name: Prof. Fabrício Olivetti de França
institute: Federal University of ABC
logo:
  - name: figs/logotipo-ufabc-abaixo.eps
  - name: figs/hal.png
cover: figs/cover.png
date: 05 February, 2024
---

## Parametric Regression Models

- Models with finite and fixed number of parameters
- Determined by prior knowledge or well established models
- Parameters are adjusted according to the available data
  - Parameters summarize the data through the model

:::::::::::::: {.columns}
::: {.column width="20%"}
\begin{equation*}
f(\mathbf{x, \theta}) = \frac{ \tikzmarknode{t1}{\highlight{red}{$\theta_{1}$}} x^3 }{ \tikzmarknode{t2}{\highlight{red}{$\theta_{2}$}} + x } + \highlight{red}{$\theta_{3}$}
\end{equation*}

:::
::: {.column width="80%"}
\begin{figure}[t!]
    \centering
    \begin{tikzpicture}\begin{axis}[domain=-10:35,legend pos=south east, width=0.7\textwidth, xlabel={temperature}, ylabel={crop yield}]
      \addplot [red, thick, mark=+] {50 - 0.3*x^3/(300+x)};
      \addplot [green, thick, mark=+] {20 + 0.1*x^3/(15+x)};
      \legend{spinach, tomato}
    \end{axis}
\end{tikzpicture}
\end{figure}
:::
::::::::::::::

## Symbolic Regression

The choice of functional form is important, so instead of using common models, we can apply Symbolic Regression to find the best function that fits the data.

:::columns

:::: {.column width="20%"}
\begin{forest}
    rounded/.style={circle, minimum size=20pt, font=\footnotesize, thick, edge={thick}, draw},
    chosen/.style={circle, minimum size=20pt, font=\footnotesize, thick, edge={thick}, fill=gray!40, draw}
    [{$+$}, for tree=rounded
      [{$*$}
        [{$x$}]
        [{$2$}]
      ]
      [{$exp$}
        [{$*$}
          [{$x$}]
          [{$x$}]
        ]
      ]
    ]
\end{forest}
::::

:::: {.column width="80%"}
\begin{figure}[t!]
    \centering
    \begin{tikzpicture}\begin{axis}[domain=-1:1,legend pos=north west, width=0.7\textwidth]
      \addplot [green, only marks] {2*x + exp(x^2) + 0.9*rand};
      \addplot [blue, thick, mark=+] {0.5*x+1.9*x^2 + 0.9};
      \addplot [red, thick, mark=+] {2*x + exp(x^2)};
      \legend{data, poly, sym reg}
    \end{axis}
\end{tikzpicture}
\end{figure}
::::

:::

## Overparameterized models

Symbolic Regression algorithms are prone to overparametrization:

$$
f(\mathbf{x, \theta}) = \theta_1 \exp{(\theta_2 x_1 + \theta_3)}
$$

- The parameters can assume different values for the same data
- Numerical issues and slow convergence of optimization
- Larger search space for memetic approaches
- Interpretation is hindered

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

## Equality Saturation

- **Equality Saturation:** used at program optimization stage of compilers
- **Equality graph:** represents multiple equivalent ASTs in parallel
- **Saturated e-graph:** contains all possible equivalences
- Currently fastest implementation uses **union-find**[^1] data structure

[^1]: Willsey, Max, et al. "Egg: Fast and extensible equality saturation." Proceedings of the ACM on Programming Languages 5.POPL (2021): 1-29.

## Equality Saturation

- An e-graph is:
    - a set of e-classes (dashed) with non-empty sets of e-nodes (solid line).
    - edge connects e-node to e-class.

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

## Future steps {.standout}

\begin{block}{}
  \begin{itemize}
    \item Investigate the benefits of reducing parameters: faster convergence, robustness, narrower uncertainty intervals.
    \item Incorporate into the evolutionary process
    \item Improve runtime
\end{itemize}

\hfill

\includegraphics[width=100pt]{figs/tobecontinued.png}
\end{block}

## Next lecture {.standout}

\begin{block}{}
  \begin{itemize}
    \item Prior Knowledge
\end{itemize}

\hfill

\includegraphics[width=100pt]{figs/tobecontinued.png}
\end{block}

## Acknowledgments
