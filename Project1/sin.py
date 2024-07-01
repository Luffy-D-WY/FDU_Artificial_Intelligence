import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.parameters = self.initialize_parameters(layer_sizes)
        self.costs = []

    def initialize_parameters(self, layer_sizes):
        parameters = {}
        for i in range(1, len(layer_sizes)):
            parameters['W' + str(i)] = np.random.randn(layer_sizes[i], layer_sizes[i-1]) * 0.01
            parameters['b' + str(i)] = np.zeros((layer_sizes[i], 1))
        return parameters

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def forward_propagation(self, X):
        caches = []
        A = X
        L = len(self.parameters) // 2

        for l in range(1, L):
            A_prev = A
            W = self.parameters['W' + str(l)]
            b = self.parameters['b' + str(l)]
            Z = W.dot(A_prev) + b
            A = self.relu(Z)
            caches.append((A_prev, Z))

        W = self.parameters['W' + str(L)]
        b = self.parameters['b' + str(L)]
        Z = W.dot(A) + b
        A = Z
        caches.append((A, Z))

        return A, caches

    def compute_cost(self, A, Y):
        m = Y.shape[1]
        cost = np.sum((A - Y) ** 2) / m
        return cost

    def backward_propagation(self, A, Y, caches):
        grads = {}
        L = len(caches)
        m = A.shape[1]
        Y = Y.reshape(A.shape)

        dA_prev = (A - Y)

        for l in reversed(range(L)):
            A_prev, Z = caches[l]
            dZ = dA_prev * (l == L-1 or self.relu_derivative(Z))
            dW = dZ.dot(A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            dA_prev = self.parameters['W' + str(l+1)].T.dot(dZ)

            grads['dW' + str(l+1)] = dW
            grads['db' + str(l+1)] = db

        return grads

    def update_parameters(self, grads, learning_rate):
        L = len(self.parameters) // 2
        for l in range(1, L+1):
            self.parameters['W' + str(l)] -= learning_rate * grads['dW' + str(l)]
            self.parameters['b' + str(l)] -= learning_rate * grads['db' + str(l)]

    def train(self, X, Y, learning_rate=0.01, num_iterations=50000):
        for i in range(num_iterations):
            A, caches = self.forward_propagation(X)
            cost = self.compute_cost(A, Y)
            self.costs.append(cost)
            grads = self.backward_propagation(A, Y, caches)
            self.update_parameters(grads, learning_rate)

            if i % 100 == 0:
                print(f"Cost after iteration {i}: {cost}")

    def predict(self, X):
        final_predictions, _ = self.forward_propagation(X)
        return final_predictions

# 数据生成
x = np.random.uniform(-np.pi, np.pi, 400).reshape(1, -1)
y = np.sin(x)
x_test=np.linspace(-np.pi,np.pi,400).reshape(1,-1)
y_test=np.sin(x_test)

# 模型参数
layer_sizes = [1, 32, 64, 1]
learning_rate = 0.01
num_iterations = 100000

# 创建并训练模型
model = NeuralNetwork(layer_sizes)
model.train(x, y, learning_rate, num_iterations)

# 预测
final_predictions = model.predict(x_test)

# 可视化
plt.figure(figsize=(10, 5))
plt.plot(x_test.flatten(), y_test.flatten(), label='True sin(x)')
plt.plot(x_test.flatten(), final_predictions.flatten(), label='NN Prediction')
plt.title("Neural Network Approximation of sin(x)")
plt.legend()
plt.show()
