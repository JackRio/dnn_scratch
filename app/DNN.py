import numpy as np
import pickle
from app.utils_dnn import load_data, clean_data, relu, sigmoid, sigmoid_backward, relu_backward

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()


class DNN:
    def __init__(self, keep_probs=None, learning_rate=0.075, epoch=1500, lamda=0.7,
                 layer_dims=None):
        self.train_x, self.test_x = clean_data(train_x_orig, test_x_orig)
        self.train_y, self.test_y = train_y, test_y

        self.layer_dims = [self.train_x.shape[0]] + layer_dims

        if keep_probs:
            self.keep_probs = keep_probs
        else:
            self.keep_probs = np.ones((len(self.layer_dims) - 1)).astype(int)

        self.learning_rate = learning_rate
        self.epoch = epoch
        self.lamda = lamda

    def initialize_param(self):
        parameters = {}
        for layer in range(1, len(self.layer_dims)):
            parameters['W' + str(layer)] = np.random.randn(
                self.layer_dims[layer],
                self.layer_dims[layer - 1]) / np.sqrt(self.layer_dims[layer - 1])  # * 0.01
            parameters['b' + str(layer)] = np.zeros((self.layer_dims[layer], 1))

            assert (parameters['W' + str(layer)].shape == (
                self.layer_dims[layer], self.layer_dims[layer - 1]))
            assert (parameters['b' + str(layer)].shape == (self.layer_dims[layer], 1))
        return parameters

    @staticmethod
    def linear_forward(A_prev, W, b, D_prev):
        Z = np.dot(W, A_prev) + b
        cache = (A_prev, W, b, D_prev)
        return Z, cache

    def linear_activation_forward(self, A_prev, W, b, keep_prob, D_prev, activation):
        Z, linear_cache = self.linear_forward(A_prev, W, b, D_prev)
        if activation == "relu":
            A, activation_cache = relu(Z)
        elif activation == "sigmoid":
            A, activation_cache = sigmoid(Z)
        else:
            raise Exception("Invalid Activation")

        # Adding dropout
        D = np.random.rand(A.shape[0], A.shape[1])
        D = (D < keep_prob).astype(int)
        A = np.multiply(A, D)
        A = A / keep_prob

        cache = (linear_cache, activation_cache)
        return A, D, cache

    def dnn_forward_prop(self, X, param):
        A = X
        D = np.random.rand(A.shape[0], A.shape[1])
        D = (D < self.keep_probs[0]).astype(int)
        A = np.multiply(A, D)
        A = A / self.keep_probs[0]

        caches = []

        for layer in range(1, (len(self.layer_dims) - 1)):
            A_prev = A
            D_prev = D
            A, D, cache = self.linear_activation_forward(A_prev, param["W" + str(layer)],
                                                         param["b" + str(layer)],
                                                         self.keep_probs[layer], D_prev, "relu")
            caches.append(cache)
        AL, D, cache = self.linear_activation_forward(A, param["W" + str(layer + 1)],
                                                      param["b" + str(layer + 1)], 1, D,
                                                      "sigmoid")
        caches.append(cache)
        return AL, caches

    @staticmethod
    def compute_cost(AL, Y):
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(np.dot(Y, np.log(AL.T)) + np.dot((1 - Y), np.log(1 - AL.T)))

        cost = np.squeeze(cost)

        return cost

    def compute_cost_with_regularization(self, param, AL, Y):
        m = Y.shape[1]
        L = len(param) // 2
        regularize_cost = 0.00
        for layer in range(L):
            regularize_cost += np.sum(np.square(param["W" + str(layer + 1)]))
        regularize_cost *= (self.lamda / (2 * m))
        cross_entropy_cost = self.compute_cost(AL, Y)

        return cross_entropy_cost + regularize_cost

    def linear_backward(self, dZ, cache, keep_prob):
        A_prev, W, b, D = cache
        m = A_prev.shape[1]

        dW = (1. / m) * np.dot(dZ, A_prev.T) + (self.lamda * W) / m
        db = (1. / m) * np.sum(dZ, keepdims=True, axis=1)
        dA_prev = np.dot(W.T, dZ)

        # Implementing dropout
        dA_prev = np.multiply(dA_prev, D)
        dA_prev /= keep_prob
        return dA_prev, dW, db

    def backward_prop(self, dAL, cache, activation, keep_prob):
        linear_cache, activation_cache = cache

        if activation == "sigmoid":
            dZ = sigmoid_backward(dAL, activation_cache)
        elif activation == "relu":
            dZ = relu_backward(dAL, activation_cache)
        else:
            raise Exception("Invalid Activation")

        dA_prev, dW, db = self.linear_backward(dZ, linear_cache, keep_prob)
        return dA_prev, dW, db

    def dnn_backward_prop(self, AL, Y, caches):
        grads = {}
        L = len(caches)
        Y = Y.reshape(AL.shape)

        dAL = - (np.divide(Y, AL) - np.divide((1 - Y), (1 - AL)))
        current_cache = caches[L - 1]
        grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = self.backward_prop(
            dAL, current_cache, "sigmoid", 1)

        for layer in reversed(range(L - 1)):
            current_cache = caches[layer]
            grads["dA" + str(layer)], grads["dW" + str(layer + 1)], grads[
                "db" + str(layer + 1)] = self.backward_prop(grads["dA" + str(layer + 1)],
                                                            current_cache, "relu",
                                                            self.keep_probs[layer])
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

            cost = self.compute_cost_with_regularization(parameters, AL, self.train_y)

            grads = self.dnn_backward_prop(AL, self.train_y, caches)

            parameters = self.update_parameters(parameters, grads)

            if not (i % 100):
                print(f"Cost after {i} iteration", cost)

        return parameters

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
    dnn = DNN(epoch=100, keep_probs=[1, 0.8, 0.6], learning_rate=0.01, lamda=0.01,
              layer_dims=[20, 10, 1])
    model_params = dnn.fit()

    with open("output/output.model", "wb") as model:
        pickle.dump(model_params, model)
    model.close()

    # with open("output/output.model", "rb") as model:
    #     model_params = pickle.load(model)
    # model.close()

    print("Train Accuracy")
    dnn.predict(dnn.train_x, dnn.train_y, model_params)

    print("Test Accuracy")
    dnn.predict(dnn.test_x, dnn.test_y, model_params)
