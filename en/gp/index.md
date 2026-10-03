---
synopsis: "A Gaussian process is a probability distribution over functions. This part builds it from Bayesian linear regression, conditions it on data, and shows how its kernel and hyperparameters encode what we believe about the unknown objective before seeing any data."
---

# Gaussian Processes {#sec-part-gp}

A Gaussian process is a probability distribution over functions: a way of saying, before any data arrive, which functions you find plausible, and of updating that belief exactly when data do arrive. It is the surrogate model in almost all Bayesian optimization, and in preferential Bayesian optimization it becomes the model of a person's hidden utility.

The first three chapters follow the life of that belief. The first builds the prior from Bayesian linear regression with ever more features, and lets you draw functions from it. The second conditions it on observations and reads the result: a posterior mean that interpolates and a posterior uncertainty that collapses where you have looked. The third asks how to choose the kernel and its hyperparameters, and why a default that works in two dimensions can quietly fail in fifty. The fourth looks under the model, at the space of functions a kernel defines, its eigenvalues and spectrum, and what a Gaussian process is as a random object: the analysis that the guarantees of @sec-part-bo rely on.

*The part needs the Gaussian conditioning formula from @sec-gaussian-conditioning and the idea of a posterior from @sec-bayesian-inference.*
