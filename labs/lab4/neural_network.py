import autograd.numpy as np
from autograd.scipy.special import logsumexp
import autograd.numpy.random as npr
from autograd import grad
from autograd.misc.optimizers import adam
import os
import sys
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# path to data
sys.path.append(os.path.join(os.getcwd(), "../../../data"))
from data.data_utils import load_dataset


# from typing import list, Tuple

def init_randn(m, n, rs=npr.RandomState(0)):
    """ init mxn matrix using small random normal
    """
    return 0.1 * rs.randn(m, n)


def init_xavier(m, n, rs=npr.RandomState(0)):
    return rs.randn(m, n) / np.sqrt(m)


def init_net_params(layer_sizes, init_fcn, rs=npr.RandomState(0)):
    """ inits a (weights, biases) tuples for all layers using the intialize function (init_fcn)"""
    return [
        (init_fcn(m, n), np.zeros(n))  # weight matrix  # bias vector
        for m, n in zip(layer_sizes[:-1], layer_sizes[1:])
    ]


def relu(x):
    return np.maximum(0, x)


def neural_net_predict(params, inputs):
    for W, b in params:
        outputs = np.dot(inputs, W) + b # general formula for transformation of a layer in neural net
        inputs = relu(outputs)
    return outputs - logsumexp(outputs, axis=1, keepdims=True)


def mean_log_like(params, inputs, targets):
    # raise NotImplementedError("mean_log_like function not completed (see Q3).")
    # print('computing mean-log-like')
    return np.mean(neural_net_predict(params, inputs) * targets) * 10

def mean_log_like2(params, inputs, targets):
    N = len(inputs)
    for W, b in params:
        outputs = np.dot(inputs, W) + b # general formula for transformation of a layer in neural net
        inputs = relu(outputs)
    # outputs = relu(outputs - logsumexp(outputs, axis=1, keepdims=True))
    log_p = 0
    for i, f_hat in enumerate(outputs):
        for j, val in enumerate(f_hat):
            if targets[i][j]:
                if (val > 0):
                    continue
                else:
                    outputs[i][j] = 0
    return log_p / N




def accuracy(params, inputs, targets):
    """ return the accuracy of the neural network defined by params
    """
    target_class = np.argmax(targets, axis=1)
    predicted_class = np.argmax(neural_net_predict(params, inputs), axis=1)
    return np.mean(predicted_class == target_class)


if __name__ == "__main__":
    ''' best so far: roughly 57% with 100 0.0004 10000 [784, 512, 10] (peaks at 50 iterations)
    some ideas:
        - two hidden layers? how many units in each?
        - probably best to tune the layer count first, then worry about getting good grad desc params
    '''
    # ----------------------------------------------------------------------------------
    # EXAMPLE OF HOW TO USE STARTER CODE BELOW
    # ----------------------------------------------------------------------------------

    # loading data
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset("mnist_small")
    # print(len(x_train))
    # initializing parameters
    layer_sizes = [784, 392,10]
    params = init_net_params(layer_sizes, init_randn)

    # setting up training parameters
    num_epochs = 30
    learning_rate = 2e-3
    batch_size = 1000 # total of 10000 datapoints

    # Constants for batching
    num_batches = int(np.ceil(len(x_train) / batch_size))
    rind = np.arange(len(x_train))
    npr.shuffle(rind)

    def batch_indices(iter):
        idx = iter % num_batches
        return rind[slice(idx * batch_size, (idx + 1) * batch_size)]

    # Define training objective
    def objective(params, iter):
        # get indices of data in batch
        idx = batch_indices(iter)
        # print(x_train[idx][0])
        return -mean_log_like(params, x_train[idx], y_train[idx])

    # Get gradient of objective using autograd.
    objective_grad = grad(objective)
    print(num_epochs, learning_rate, batch_size, layer_sizes)
    print("     Epoch     |    Train accuracy  |       Test accuracy  ")

    # Dictionary to store train/val history
    opt_history = {
        "train_nll": [],
        "val_nll": [],
    }
    global g_params
    g_params = None

    def callback(params, iter, gradient):
        if iter % num_batches == 0:
            # record training & val. accuracy every epoch
            opt_history["train_nll"].append(-mean_log_like(params, x_train, y_train))
            opt_history["val_nll"].append(-mean_log_like(params, x_valid, y_valid))
            train_acc = accuracy(params, x_train, y_train)
            val_acc = accuracy(params, x_valid, y_valid)
            print("{:15}|{:20}|{:20}".format(iter // num_batches, train_acc, val_acc))
            global g_params
            g_params = params

    # We will optimize using Adam (a variant of SGD that makes better use of
    # gradient information).
    opt_params = adam(
        objective_grad,
        params,
        step_size=learning_rate,
        num_iters=num_epochs * num_batches,
        callback=callback,
    )
    # Plotting the train and validation negative log-likelihood
    plt.plot(opt_history["train_nll"], label='training')
    plt.plot(opt_history["val_nll"], label = 'validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # plt.show()
    plt.cla()
    target_class = np.argmax(y_test, axis=1)
    predicted_class = np.argmax(neural_net_predict(g_params, x_test), axis=1)
    cm = confusion_matrix(predicted_class, target_class)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.show()


