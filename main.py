# from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml
from skopt.space import Real

from bo import BayesianOptimization

if __name__ == "__main__":
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist['data'], mnist['target']
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

    # model = SGDClassifier(loss='log', random_state=42, learning_rate='constant', eta0=0.1)
    model = MLPClassifier(hidden_layer_sizes=(512, 256), activation='logistic', solver='sgd', batch_size=128, learning_rate='constant', learning_rate_init=0.1)
    search_space = [Real(1e-7, 0.1, name='learning_rate_init')]

    bo = BayesianOptimization(model, search_space, X_train, y_train)
    bo.run()