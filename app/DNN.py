import numpy as np
from app.utils_dnn import load_data, clean_data, relu, sigmoid, sigmoid_backward, relu_backward

np.random.seed(1)
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()


class DNN:
    def __init__(self, learning_rate=0.05, epoch=2500):
        self.train_x, self.test_x = clean_data(train_x_orig, test_x_orig)
        self.train_y, self.test_y = train_y, test_y

        # 4-layer model (Input + 3 Hidden + Output)
        self.layer_dims = [self.train_x.shape[0], 20, 10, 5, 1]
        self.learning_rate = learning_rate
        self.epoch = epoch

    def initialize_param(self):
        parameters = {}
        for layer in range(1, len(self.layer_dims)):
            parameters['W' + str(layer)] = np.random.randn(self.layer_dims[layer],
                                                           self.layer_dims[layer - 1]) * 0.01
            parameters['b' + str(layer)] = np.zeros((self.layer_dims[layer], 1))
        return parameters

    @staticmethod
    def linear_forward(A_prev, W, b):

        Z = np.dot(W, A_prev) + b
        cache = (A_prev, W, b)
        return Z, cache

    def linear_activation_forward(self, A_prev, W, b, activation):
        Z, linear_cache = self.linear_forward(A_prev, W, b)
        if activation == "relu":
            A, activation_cache = relu(Z)
        elif activation == "sigmoid":
            A, activation_cache = sigmoid(Z)
        else:
            raise Exception("Invalid Activation")

        cache = (linear_cache, activation_cache)
        return A, cache

    def dnn_forward_prop(self, X, param):
        A = X
        caches = []

        for layer in range(1, (len(self.layer_dims) - 1)):
            A_prev = A
            A, cache = self.linear_activation_forward(A_prev, param["W" + str(layer)],
                                                      param["b" + str(layer)], "relu")
            caches.append(cache)
        AL, cache = self.linear_activation_forward(A, param["W" + str(layer + 1)],
                                                   param["b" + str(layer + 1)], "sigmoid")
        caches.append(cache)
        return AL, caches

    @staticmethod
    def compute_cost(AL, Y):
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(np.dot(Y, np.log(AL.T)) + np.dot((1 - Y), np.log(1 - AL.T)))

        cost = np.squeeze(cost)

        return cost

    @staticmethod
    def linear_backword(dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = (1 / m) * np.dot(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, keepdims=True, axis=1)
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db

    def backword_prop(self, dAL, cache, activation):
        linear_cache, activation_cache = cache

        if activation == "sigmoid":
            dZ = sigmoid_backward(dAL, activation_cache)
        elif activation == "relu":
            dZ = relu_backward(dAL, activation_cache)
        else:
            raise Exception("Invalid Activation")

        dA_prev, dW, db = self.linear_backword(dZ, linear_cache)
        return dA_prev, dW, db

    def dnn_backword_prop(self, AL, Y, caches):
        grads = {}
        L = len(caches)
        Y = Y.reshape(AL.shape)

        dAL = - (np.divide(Y, AL) - np.divide((1 - Y), (1 - AL)))
        current_cache = caches[L - 1]
        grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = self.backword_prop(
            dAL, current_cache, "sigmoid")
        for layer in reversed(range(L - 1)):
            current_cache = caches[layer]

            grads["dA" + str(layer)], grads["dW" + str(layer + 1)], grads[
                "db" + str(layer + 1)] = self.backword_prop(grads["dA" + str(layer + 1)],
                                                            current_cache, "relu")
        return grads

    def update_parameters(self, param, grads):
        L = len(self.layer_dims) - 1

        for layer in range(L):
            param["W" + str(layer + 1)] -= grads["dW" + str(layer + 1)] * self.learning_rate
            param["b" + str(layer + 1)] -= grads["db" + str(layer + 1)] * self.learning_rate
        return param

    def fit(self):
        parameters = self.initialize_param()

        for i in range(self.epoch):
            AL, caches = self.dnn_forward_prop(self.train_x, parameters)

            cost = self.compute_cost(AL, self.train_y)

            grads = self.dnn_backword_prop(AL, self.train_y, caches)

            param = self.update_parameters(parameters, grads)

            if not (i % 100):
                print("Cost", cost)

        return param

    def predict(self, X, y, param):
        m = X.shape[1]
        p = np.zeros((1, m))

        # Forward propagation
        probas, caches = self.dnn_forward_prop(X, param)

        # convert probas to 0/1 predictions
        for i in range(0, probas.shape[1]):
            if probas[0, i] > 0.5:
                p[0, i] = 1
            else:
                p[0, i] = 0

        print("Accuracy: " + str(np.sum((p == y) / m)))

        return p


if __name__ == "__main__":
    dnn = DNN()
    parameters = dnn.fit()

    print("Train Accuracy")
    dnn.predict(dnn.train_x, dnn.train_y, parameters)

    print("Test Accuracy")
    dnn.predict(dnn.test_x, dnn.test_y, parameters)
