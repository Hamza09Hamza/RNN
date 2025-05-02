import numpy as np

class LogisticRegression:
    def __init__(
        self,
        alpha=0.01,
        iterations=1000,
        use_l2=False,
        lambda_=0.01,
        use_decay=False,
        decay=0.0,
        early_stopping=False,
        tol=1e-4
    ):
        self.alpha = alpha
        self.iterations = iterations
        self.use_l2 = use_l2
        self.lambda_ = lambda_
        self.use_decay = use_decay
        self.decay = decay
        self.early_stopping = early_stopping
        self.tol = tol
        self.theta = None
        self.cost_history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_cost(self, X, y, theta):
        m = len(y)
        h = self.sigmoid(X @ theta)
        epsilon = 1e-8  # To avoid log(0)
        cost = -1/m * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
        if self.use_l2:
            reg = (self.lambda_ / (2 * m)) * np.sum(theta[1:] ** 2)
            cost += reg
        return cost

    def gradient_descent(self, X, y, theta):
        m = len(y)
        cost_history = []

        for i in range(self.iterations):
            h = self.sigmoid(X @ theta)
            gradient = (X.T @ (h - y)) / m

            if self.use_l2:
                gradient[1:] += (self.lambda_ / m) * theta[1:]

            learning_rate = self.alpha
            if self.use_decay:
                learning_rate = self.alpha / (1 + self.decay * i)

            theta -= learning_rate * gradient
            cost = self.compute_cost(X, y, theta)
            cost_history.append(cost)

            if self.early_stopping and i > 0:
                if abs(cost_history[-2] - cost) < self.tol:
                    print(f"Early stopping at iteration {i}")
                    break

        return theta, cost_history

    def fit(self, X, y):
        X = np.c_[np.ones((X.shape[0], 1)), X]
        theta = np.zeros(X.shape[1])
        self.theta, self.cost_history = self.gradient_descent(X, y, theta)
        return self

    def predict_proba(self, X):
        X = np.c_[np.ones((X.shape[0], 1)), X]
        return self.sigmoid(X @ self.theta)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
