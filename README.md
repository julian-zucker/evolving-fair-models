# Evolving fair models

Here's the idea:

- There are multiple ways of computing fairness, and you also care about predictive accuracy
- Inducing a model is therefore a multi-objective optimization problem
- Evolutionary computing is good at solving multi-objective optimization problems

So, I'm going to try to use evolutionary computing to evolve fair models.

What type of model?
- Decision trees, because trees are a well-known "solution type" in EC.

What fairness metrics?
- AIF360 provides a bunch, let's use those

Also, as a side benefit of creating a Pareto frontier of models that have various tradeoffs in accuracy/fairness, I can explore the relationships between different notions of fairness and predictive accuracy. I can even look at the tradeoffs between different notions of fairness! That last one ought to be interesting. 

