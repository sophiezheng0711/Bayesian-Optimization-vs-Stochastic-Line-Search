# Bayesian-Optimization-vs-Stochastic-Line-Search

As the name suggests, we are comparing BO and SLS on classic classification problems like MNIST and CIFAR100.

## Files

```
    # runs BO on MNIST
    main.py
    # implements bayesian optimization that takes a model, a search space, X data, and y data
    bo.py
```

# Bayesian Optimization (BO)

## Key Packages

We used `scikit-optimize` (skopt) for our Bayesian Optimization implementation. We mainly use the `skopt.gp_minimize` function with the EI (expected improvement) acquisition function.

# Stochastic Line Search (SLS)
