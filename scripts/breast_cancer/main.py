import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Perceptron:

    def __init__(self, eta=0.10, epochs=50, is_verbose=False):

        self.eta = eta
        self.epochs = epochs
        self.is_verbose = is_verbose

    def predict(self, x):

        ones = np.ones((x.shape[0], 1))
        x_1 = np.append(x.copy(), ones, axis=1)
        return np.where(self.get_activation(x_1) > 0, 1, -1)

    def get_activation(self, x):
        activation = np.dot(x, self.w)
        return activation

    def fit(self, X, y):

        self.list_of_errors = []

        ones = np.ones((X.shape[0], 1))
        X_1 = np.append(X.copy(), ones, axis=1)

        self.w = np.random.rand(X_1.shape[1])

        for e in range(self.epochs):

            error = 0

            activation = self.get_activation(X_1)
            delta_w = self.eta * np.dot((y - activation), X_1)
            self.w += delta_w

            error = np.square(y - activation).sum()/2.0
            self.list_of_errors.append(error)

            if (self.is_verbose):
                print("Epoch: {}, weights: {}, number of errors: {}".format(e, self.w, error))

diag = pd.read_csv('../../data/breast_cancer.csv')

X = diag[['area_mean', 'area_se', 'texture_mean', 'concavity_worst', 'concavity_mean']]
y = diag['diagnosis']
y = y.apply(lambda x: 1 if x=='M' else -1)

# perceptron = Perceptron(0.0000000001, 100, True)
# perceptron.fit(X, y)
#
# plt.scatter(range(perceptron.epochs), perceptron.list_of_errors)
# plt.show()

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
perceptron = Perceptron(0.001, 100, True)
perceptron.fit(X, y)

plt.scatter(range(perceptron.epochs), perceptron.list_of_errors)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y)
perceptron.fit(X_train, y_train)
y_pred = perceptron.predict(X_test)

good = y_test[y_test == y_pred].count()
total = y_test.count()

accuracy = good / total * 100

print('Accuracy: {} %'.format(accuracy))
