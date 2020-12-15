# Bayesian-Optimization-vs-Stochastic-Line-Search

As the name suggests, we are comparing BO and SLS on classic classification problems like MNIST and CIFAR100.

## Run Instructions

To run Bayesian Optimization related experiments, run the following:
```
    python3 bo_main.py -d mnist
    python3 bo_main.py -d mushroom
```

To run Stochastic Line Search related experiments, run the following:
```
    python3 sls_torch.py
```

## Files

```
    # runs BO on MNIST/mushroom
    bo_main.py
    # implements bayesian optimization that takes a model, a search space, X data, and y data
    bo.py
    # SLS optimizer implementation in PyTorch
    sls_optimizer.py
    # runs SLS on MNIST/mushroom
    sls_torch.py
```

# Bayesian Optimization (BO)

## Key Packages

We used `scikit-optimize` (skopt) for our Bayesian Optimization implementation. We mainly use the `skopt.gp_minimize` function with the EI (expected improvement) acquisition function.

# Stochastic Line Search (SLS)
