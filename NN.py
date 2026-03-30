import numpy as np

class MLP:

    def __init__(self, input_dim, hidden_dim):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, 1) * 0.1
        self.b2 = np.zeros((1, 1))
        self.part1 = np.empty(shape=(hidden_dim, 1))
        self.activated1 = np.empty(shape=(hidden_dim, 1))
        self.part2 = np.empty(shape=(hidden_dim, 1))
        self.forward_out = np.empty(shape=(hidden_dim, 1))
        self.dW2 = np.empty(shape=(hidden_dim, 1))
        self.db2 = 0
        self.dW1 = np.empty(shape=(input_dim, 1))
        self.db1 = 0

    def forward(self, x):
        self.part1 = np.dot(x, self.W1) + self.b1
        self.activated1 = np.maximum(0.01*self.part1, self.part1)

        self.part2 = np.dot(self.activated1, self.W2) + self.b2
        self.forward_out = np.maximum(0.01*self.part2, self.part2)

        return self.forward_out

    def backward(self, x, y, output):
        error = output - y.reshape(-1, 1)
        # print(f"error: {error}")

        self.dW2 = np.dot(self.activated1.T, error)
        self.db2 = np.sum(error, axis=0, keepdims=True)

        d_relu = np.ones_like(self.activated1)
        mask = self.activated1 <= 0
        d_relu[mask] = 0.01
        delta_hidden = np.dot(error, self.W2.T) * d_relu

        self.dW1 = np.dot(x.T, delta_hidden)
        self.db1 = np.sum(delta_hidden, axis=0, keepdims=True)

def update_weights(model, learning_rate):
    model.W1 -= learning_rate * model.dW1
    model.b1 -= learning_rate * model.db1
    model.W2 -= learning_rate * model.dW2
    model.b2 -= learning_rate * model.db2