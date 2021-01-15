import numpy as np


class LogisticRegression:
    def __init__(self, X_train, y_train, learning_rate=1, num_iterations=100, print_costs=True):
        self.X_train = X_train
        self.y_train = y_train
        self.learning_rate = learning_rate
        self.w = np.zeros((self._feature_dimension, 1))
        self.b = 0
        self.dw = 0
        self.db = 0
        self.num_iterations = num_iterations
        self.cost = None
        self.print_costs = print_costs

        self.train_data()



    @classmethod
    def sigmoid(self, z):
        s = 1 / (1 + np.exp(-z))
        return s

    @property
    def _feature_dimension(self):
        return self.X_train.shape[0]

    def train_data(self):
        self.optimize()

    def propagate(self):
        m = self.X_train.shape[1]

        # Forward propagation
        z = np.dot(self.w.T, self.X_train) + self.b
        A = self.sigmoid(z)
        self.cost = -(1 / m) * np.sum(self.y_train * np.log(A) + ((1 - self.y_train) * np.log(1 - A)))

        # Backward propagation
        self.dw = (1 / m) * np.dot(self.X_train, (A - self.y_train).T)
        self.db = (1 / m) * np.sum(A - self.y_train)

    def optimize(self):
        for i in range(self.num_iterations):

            self.propagate()

            self.w = self.w - np.dot(self.learning_rate, self.dw)
            self.b = self.b - np.dot(self.learning_rate, self.db)


    def predict(self, X):
        m = X.shape[1]
        y_pred = np.zeros((1, m))
        w = self.w.reshape(X.shape[0], 1)

        z = np.dot(w.T, self.X_train) + self.b
        A = self.sigmoid(z)

        for i in range(A.shape[1]):
            y_pred[0][i] = 0 if A[0][i] <= 0.5 else 1

        return y_pred