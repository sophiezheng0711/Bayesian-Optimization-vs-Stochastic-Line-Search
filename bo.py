import numpy as np
from matplotlib import pyplot as plt
from warnings import catch_warnings, simplefilter
from sklearn.model_selection import cross_val_score
from skopt.utils import use_named_args
from skopt import gp_minimize


class BayesianOptimization():
    def __init__(self, model, search_space, X, y, epochs):
        self.model = model
        self.search_space = search_space
        self.X = X
        self.y = y
        self.epochs = epochs

    def eval(self):
        @use_named_args(self.search_space)
        def evaluate_model(**params):
            self.model.set_params(**params)
            # calculate 5-fold cross validation
            result = cross_val_score(self.model, self.X, self.y, cv=5, n_jobs=-1, scoring='accuracy')
            # calculate the mean of the scores
            estimate = np.mean(result)
            print(1.0 - estimate)
            return 1.0 - estimate
        return evaluate_model
    
    def run(self):
        with catch_warnings():
            simplefilter('ignore')
            result = gp_minimize(self.eval(), self.search_space, acq_func='EI', n_calls=self.epochs)
            result_etas = np.array([x[0] for x in result.x_iters])
            result_accs = np.array([1 - x for x in result.func_vals])
            print('Best Accuracy: %.3f' % (1.0 - result.fun))
            print('Best Parameters: eta0=%.5e' % (result.x[0]))
            plt.scatter(result_etas, result_accs)
            plt.show()
