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

### Gameplan
1. Choose programming language/tools. There are two real options here: Python and Scala. Python has the benefit of DEAP and AIF360 being implemented in it. Scala would allow us to use Evvo. Much as I dislike python, it may be the better choice here, for the DEAP/AIF360.
2. Choose the datasets we'll work on.  This is probably a good time to review the literature and pick datasets that other people commonly use.
3. Write a rough outline of the paper, enough that we understand where our results will fit into our arguments/explorations about fairness.s
4. Once that's settled, we'll have to write a bunch of code:
- create a data representation for a decision tree that we can evolve
- load the datasets
- on each dataset, create an objective/fitness function for each fairness metric
- run evolution with objectives as accuracy/one fairness metric
- produce the pair-wise tradeoffs between each pair of fairness metrics, at a specific point. Come up with a principled way of choosing the point that the fairness metrics crossover.
5. Now, we'll have a bunch of data. We should inspect it by hand to get a rough notion of what we'll visualize.
6. Make the data visualizations.
7. Put the davis in the paper.
8. Finish writing the paper.
9. Find a publication venue.


Things to keep in mind:
- we'll want to publish this code afterwards, so we should make sure we keep good code quality/comment everything
- reproducible results are key
- we'll probably want to rope in a professor for academic clout - maybe after step 5, although I suspect that step 4 will take most of our time
- this can be publishable as a philsophy paper (ML fairness as focus) or CS paper (use of evolutionary computing, specifically, for ML fairness), so keep both in mind as we're exploring


