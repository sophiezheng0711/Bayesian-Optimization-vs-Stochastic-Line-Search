import numpy as np
from matplotlib import pyplot
from warnings import catch_warnings, simplefilter
from sklearn.model_selection import cross_val_score
from skopt.space import Integer
from skopt.utils import use_named_args
from skopt import gp_minimize

from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier


class BayesianOptimization():
    def __init__(self, model, search_space):
        self.model = model
        self.search_space = search_space

    def eval(self):
        @use_named_args(self.search_space)
        def evaluate_model(**params):
            model.set_params(**params)
            # calculate 5-fold cross validation
            result = cross_val_score(model, X, y, cv=5, n_jobs=-1, scoring='accuracy')
            # calculate the mean of the scores
            estimate = np.mean(result)
            return 1.0 - estimate
        return evaluate_model
    
    def run(self):
        with catch_warnings():
            simplefilter('ignore')
            result = gp_minimize(self.eval(), self.search_space, acq_func='EI')
            # to be changed, since this print only works for the example test
            print('Best Accuracy: %.3f' % (1.0 - result.fun))
            print('Best Parameters: n_neighbors=%d, p=%d' % (result.x[0], result.x[1]))
        

if __name__ == "__main__":
    # make test data. To be replaced with real data.
    X, y = make_blobs(n_samples=500, centers=3, n_features=2)
    model = KNeighborsClassifier()
    # define search space. In this example, there are two hyperparameters we are trying to optimize.
    # in addition, this is an integer space. We could also use skopt.space.space.Real, for real parameters.
    search_space = [Integer(1, 5, name='n_neighbors'), Integer(1, 2, name='p')]

    bo = BayesianOptimization(model, search_space)
    bo.run()