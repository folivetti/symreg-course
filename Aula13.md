---
title: Model Validation
author:
  - name: Prof. Fabrício Olivetti de França
institute: Federal University of ABC
logo:
  - name: figs/logotipo-ufabc-abaixo.eps
  - name: figs/hal.png
cover: figs/cover.png
date: 05 February, 2024
---

# Model Validation

## Model Validation
\justifying

As already stressed throughout this course, there are three main approaches for nonlinear regression:

- Using an overparameterized generic model (opaque model).
- Manually crafting the nonlinear model.
- Using Symbolic Regression to find a nonlinear model with as few parameters as possible.

## Model Validation
\justifying

While crafting the model using first principles, you may have some properties that you want to enforce into your model, either because of some requirements or from a prior knowledge about the behavior of the system.

In this situation, the practitioner can enforce those using their own expertise.

## Model Validation
\justifying

For example, due to EU regulations[^1], the practitioner will create a model that will allow them to debug how the output is generate in a clear manner. Also, they may want to ensure fairness in the predictions.

[^1]: (https://www.europarl.europa.eu/news/en/press-room/20240308IPR19015/artificial-intelligence-act-meps-adopt-landmark-law)[https://www.europarl.europa.eu/news/en/press-room/20240308IPR19015/artificial-intelligence-act-meps-adopt-landmark-law]

## Model Validation
\justifying

This is usually a problem for opaque models that are often hard to debug and not flexible enough to enforce some propeties of interest.

In the current literature, there are some techniques that can extract information from opaque models to have a better understanding. But this may not be enough in practice.

## Model Validation
\justifying

With the *vanilla* symbolic regression, you have the possibility of finding a model that meets all your requirements. To increase the probability of finding the correct model, you'll need:

- Noiseless data.
- Representative data.
- Luck \twemoji{four leaf clover}
- A well calibrated SR algorithm.

## Model Validation
\justifying

With the *vanilla* symbolic regression, you have the possibility of finding a model that meets all your requirements. To increase the probability of finding the correct model, you'll need:

- Noiseless data.
- Representative data.
- Luck \twemoji{four leaf clover}
- A well calibrated SR algorithm.

We can only afford the last one!

## Model Validation
\justifying

Another important motivation for model validation is that, depending on the hyper-parameters, the SR algorithm can favor large and overparameterized models that will have a high goodness-of-fit without the remaining desiderata.

## Example of desiderata
\justifying

Some example of objectives beyond the goodness-of-fit[^2] are:

- The ability to understand and explain model behavior
- Scientific plausibility of the model
- Whether the model is generalizable and capable of extrapolation
- Boundedness and safe operation under all circumstances
- Efficiency of calculating predictions or computational effort required for
training the model

[^2]: Gabriel Kronberger, Bogdan Burlacu, Michael Kommenda, Stephan M. Winkler, and Michael Affenzeller. Symbolic Regression. tbr.

## Example of desiderata
\justifying

Besides those, we may also want a model that:

- Ensures a fair inference to different classes of the sample.
- Behaves according to pre-established norms.

## Ability to understand and explain model behavior
\justifying

In the beggining of the course, it was clear that a linear model is easy to understand:

- With every unitary change in $x$ we observe a change proportional to $\beta$ in the outcome.
- Even if we have a linear model with non-linear features, they can have physical meaning. E.g., $v = s / t$, the inverse interaction of displacement and time gives us the average velocity.

## Ability to understand and explain model behavior
\justifying

When we have a nonlinear regression model, these interepretations are not as straightforward:

\begin{empheq}[box=\mybox]{align*}
f(x; \theta) = \frac{\theta_1 x}{\theta_2 + x},
\end{empheq}

The association between the input variable and the outcome is not easily understood.

## Ability to understand and explain model behavior
\justifying

We can try to understand the behavior with a plot for different values of $\theta$:

\begin{tikzpicture}
\begin{axis}[legend pos=outer north east, domain=0:5,samples=1000]
\addplot [very thick,red] {2*x/(0.25 + x)}; \addlegendentry{$\theta = [2, 0.25]$}
\addplot [very thick,blue] {5*x/(0.25 + x)}; \addlegendentry{$\theta = [5, 0.25]$}
\addplot [very thick,green] {2*x/(0.55 + x)}; \addlegendentry{$\theta = [2, 0.55]$}
\addplot [very thick,black] {5*x/(0.55 + x)}; \addlegendentry{$\theta = [5, 0.55]$}
\end{axis}
\end{tikzpicture}

## Ability to understand and explain model behavior
\justifying

::::columns
:::column
\begin{tikzpicture}
\begin{axis}[domain=0:5,samples=1000]
\addplot [very thick,red] {2*x/(0.25 + x)};
\addplot [very thick,blue] {5*x/(0.25 + x)};
\addplot [very thick,green] {2*x/(0.55 + x)};
\addplot [very thick,black] {5*x/(0.55 + x)};
\end{axis}
\end{tikzpicture}
:::
:::column
- This model has a saturation value close to $\theta_1$
- The higher the value of $\theta_2$, the slower the speed to reach the saturation
- When $x = \theta_2$, $f(x; \theta) = 0.5 \theta_1$, so it is the point where we reach about half saturation
- There is an undefined behavior at $x = -\theta_2$
::::columns

## Ability to understand and explain model behavior
\justifying

Having the context of the model can help gain additional insights. This particular model can represent the **Michaelis–Menten kinetics** that describes the reaction rate ($f(x; \theta)$) to the concentration of a substrate ($x$).

Knowing the phyisical meanings of $\theta$ will give us insight when fitting this model for different enzymes.

We can see that, once we contextualize the model and add expert knowledge, we can gain insights from nonlinear models as well, as long as their parameters are meaningful in our context (thus, minimize the number of parameters is desired).

## Ability to understand and explain model behavior
\justifying

In short, inspecting the model for the ability of understanding and explaining can be done by:

- Contextualizing the model
- Applying expert knowledge
- Plotting the behavior of the function with different parameter values

Additional tools will be given in later lectures when we talk about explainability.

## Scientific plausibility of the model
\justifying

Related to the previous desiderata, scientific plausibility refers to whether the model:

- Behaves similarly to the observed phenomena.
- Is correct w.r.t. a dimensional analysis (or whether all meta-features are dimensionless)
- Possesses a phyisical meaning
- Does not misbehave

This can be inspected through visual plots and expert knowledge.

## Whether the model is generalizable and capable of extrapolation
\justifying

The SR model is fitted on a limited data set that does not necesseraly captures the whole domain.

% replace this
\begin{tikzpicture}\begin{axis}[domain=-15:40, ymin=-1.5, ymax=1.5, legend pos=south east,scale only axis=true,width=0.4\textwidth,height=0.3\textwidth,tick label style={font=\tiny},label style={font=\tiny}, legend style={font=\tiny}]
      \addplot[domain=-15:40, samples=1000, olive, thick]{0.191257247217522*sin(deg(5.5016*x)) + 1.0011*tanh(0.2641*x + 0.0458257569495584*sqrt(abs(x))) - 0.0247};
      \addplot[domain=-15:40, samples=1000, red, thick]{0.333333333333333*x/sqrt(0.111111111111111*x*2 + 1)};
      \addplot[domain=-15:40, samples=1000, blue, thick]{0.989484*tanh(0.274847*x + 0.0440621) - 0.00121047};
      \addplot[domain=-15:40, samples=1000, black] gnuplot[id=erf]{erf(0.22*x) + 0.17*sin(5.5*x)};
      \addplot +[mark=none, dashed, black] coordinates {(15, -1.5) (15, 1.5)};
      \addplot +[mark=none, dashed, black] coordinates {(40, -1.5) (40, 1.5)};
      \legend{operon-easy, uDSR-medium, QLattice-hard, ground-truth}
    \end{axis}
\end{tikzpicture}

## Whether the model is generalizable and capable of out-of-domain extrapolation
\justifying

To verify whether the SR model is well behaved outside the domain we can:

- Plot the model outside the training range (works well up to $2$ dimensions)
- Assert some desirable properties (monotonicity, concavity, periodicity; but not easy to assert)
- Collect additional points outside the training domain (may not be possible or it may cost too much)

This is still an open problem and the solution depends on what kind of information we have available.

## Boundedness and safe operation under all circumstances
\justifying

The generated model may be a partial function or misbehave at certain extremal points.

For example, if we have a division $f(x)/g(x)$, it will be undefined at $g(x) = 0$. This may create a problem if
we are using this model in practice. What should we return if that happens?

Sometimes we can observe an exponential growth at the extrema of the domain of $x$, this can reflect on an increased error of the model predictions close to those points.

## Boundedness and safe operation under all circumstances
\justifying

First of all, we must confirm if such model is accceptable:

- Is there any value of $x$ in which $f(x; \theta)$ is undefined or unbounded?
- Even if it is bounded, does it show an undesirable behavior (e.g., exponential growth)?
- Do we have some means to treat such errors?
- If we want to fit this same model into different data, will it misbehave for a certain $(x; \theta)$?

## Boundedness and safe operation under all circumstances
\justifying

One solution is to replace the operator set with protected operators:

- Returning a default value on error (e.g., division by $0$ will return $1$)
- Using composition of operators (e.g., replace $log$ by $log \circ abs$)
- Using alternative operators that behave similarly to the original (e.g., $AQ(x,y) = x/\sqrt{1 + x^2} \approx x/y$)

## Boundedness and safe operation under all circumstances
\justifying

We can also evaluate the partiality of the expression using interval arithmetic if we know the domain of $x$.
In this case, we can penalize or even discard functions that are unsafe for that particular domain.

This can be a good compromise as we can still use the original operators but do not discard them entirely.

## Efficiency of calculating predictions or computational effort required for training the model
\justifying

In some situations, the efficiency of the prediction or even to obtain the fitted model is important:

- Limited time or computational budget
- Real-time system
- Data set is too large, making the evaluation of large expressions too costly

## Efficiency of calculating predictions or computational effort required for training the model
\justifying

These objectives influence the choice of operator set (addition costs much less than calculating a trigonometric function), the limits of the expression size, the algorithm implementation, and even the search algorithm.

In some situations a populational search may not be the best choice, even with the cost of generating a worse solution.

## Ensures a fair inference to different classes of the sample.
\justifying

When the model can have a social impact, we need to ensure that the model will not commit a prediction error that negatively affects people's life:

- Arrest someone by mistake
- Misdiagnose a patient

## Ensures a fair inference to different classes of the sample.
\justifying

Even worse if those mistakes occur due to bias in the data. We have alread cases of:

- ML models increasing the prediction of fellony for black and latin american people (https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing)
- Learning neo-nazi speech in a conversation bot (https://www.technologyreview.com/s/610634/microsofts-neo-nazi-sexbot-was-a-great-lesson-for-makers-of-ai-assistants/
)

## Ensures a fair inference to different classes of the sample.
\justifying

Being unfair is not a fault of SR algorithm, specially this is not a (easily) measurable obejctive.

But, depending on the generated model, SR can at least facilitate the detection of any unfairness. As such, the practitioner should pay attention certain protected variables: genre, etnicity, age, home address, and any other variable that correlates to those

We can eliminate these variables from the dataset before generating the model. A better solution is to inspect the model after it is generated to see how it uses such variable.

## Ensures a fair inference to different classes of the sample.
\justifying

Example. given a model that suggest treatment for a patient with a certain disease. We should investigate the behavior of the model for certain misbehaviors:

- Holding everything equal, if we change race in the input variable, does it change the recommendation for better treatments?
- Does the dosage of a certain treatment varies with different race? If it does, is this variation explained and supported by any study? (e.g., a certain genotype is more resistant to treatment)

## Ensures a fair inference to different classes of the sample.
\justifying

To alleviate this issue during the model search, we can incorporate fairness measures into the objective by either using multi-objective or applying a penalization strategy.

We can measure fairness and equity as:

- Statistical parity: each group has a distribution of responses proportional by their representativiness
- Inequality impact: whether the average response for two groups are approximately the same
- Opportunity equality: whether all groups have the same probability of a positive outcome
- Calibration: whether the false positive rates are equal among the groups
- Counterfactual equity: given a positive outcome, this is unaffected when changing the protected variables


## Behaves according to pre-established norms.
\justifying


## Visual tools
\justifying

## Splitting Data
\justifying

## Analytical Analysis
\justifying

https://mindfulmodeler.substack.com/p/how-to-get-from-evaluation-to-final

## Next lecture {.standout}

\begin{block}{}
  \begin{itemize}
    \item Model Selection
\end{itemize}

\hfill

\includegraphics[width=100pt]{figs/tobecontinued.png}
\end{block}

## Acknowledgments
