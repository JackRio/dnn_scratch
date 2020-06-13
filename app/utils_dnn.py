import numpy as np
import math
import h5py


def relu(Z):
    A = np.maximum(0, Z)
    return A, Z


def sigmoid(Z):
    Z = np.array(Z)

    A = 1 / (1 + np.exp(-Z))
    return A, Z


def clean_data(train_orig, test_orig):
    # Train and Test samples
    m_train = train_orig.shape[0]
    m_test = test_orig.shape[0]

    train_x_flatten = train_orig.reshape(m_train, -1).T
    test_x_flatten = test_orig.reshape(m_test, -1).T

    # Normalizing
    train_x = train_x_flatten / 255.
    test_x = test_x_flatten / 255.

    return train_x, test_x


def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ


def sigmoid_backward(dA, cache):
    Z = cache

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    assert (dZ.shape == Z.shape)

    return dZ


def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def generate_mini_batches(X, Y, batch_size, seed=0):
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []

    perm = np.random.permutation(m)
    X_shuffled = X[:, perm]
    Y_shuffled = Y[:, perm]

    num_of_mini_batches_complete = math.floor(m / batch_size)
    for i in range(num_of_mini_batches_complete):
        mini_batch_X = X_shuffled[:, i * batch_size:(i + 1) * batch_size]
        mini_batch_Y = Y_shuffled[:, i * batch_size:(i + 1) * batch_size].reshape(Y.shape[0],
                                                                                  batch_size)
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % batch_size != 0:
        mini_batch_X = X_shuffled[:, num_of_mini_batches_complete * batch_size:]
        mini_batch_Y = Y_shuffled[:, num_of_mini_batches_complete * batch_size:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches
