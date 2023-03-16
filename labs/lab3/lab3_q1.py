from data.data_utils import load_dataset # include copy of data folder in submission zip
import numpy as np
from time import time
import math
from matplotlib import pyplot as plt

# TODO: test for different values of learning rate, beta
# TODO: graph results

np.random.seed = 1006842534

class graddesc:
    '''
    class stores dataset to avoid multiple imports, and also stores hyperarameters
    related to the search algorithm. The class can be specified to perform any of the
    three variations of gradient descent with any desired hyperparamters
    '''
    def __init__(self, dataset, iter = 100, l_rate = 0, beta = 0, batch = 0, track_err = False):
        self.x_train, self.x_valid, self.x_test, self.y_train, self.y_valid, self.y_test = load_dataset(dataset)
        np.random.shuffle(self.x_train), np.random.shuffle(self.y_train)
        self.x_train, self.y_train = self.x_train[:1000], self.y_train[:1000]
        print(np.shape(self.x_train.T.dot(self.y_train)))
        self.iter = iter
        self.lr = l_rate
        self.beta = beta
        self.batch = batch
        self.w = np.zeros((np.shape(self.x_train[0])[0], 1))
        self.track = track_err
        self.losses = [0] * self.iter
        # if track_err, then we will measure at each step
        # the least squares loss so that we can plot it later
        # note: for the plot, we should also plot the absolute minimum for the ls loss due to 

    
    def df_dw(self):
        # determine gradient for current value of w
        grad = - np.matmul(self.x_train.T, self.y_train) - (self.x_train.T.dot(self.x_train.dot(self.w)))
        return grad

    def run_gd(self):
        # run the appropriate variation of grad desc
        if self.beta != 0:
            self.run_momentum()
        elif self.batch != 0:
            self.run_stoch()
        else:
            self.run_full()
    
    def run_momentum(self):
        # run gradient descent with momentum, using the Beta and batch size specified upon declaration
        pass

    def run_stoch(self):
        # run sotchastic gradient descent, using the batch size specified upon declaration
        pass

    def run_full(self):
        # run a full gradient descent using all 1000 training points
        for i in range(self.iter):
            self.w = self.w  - self.lr * self.df_dw()
            if self.track:
                self.losses[i] = 0 # TODO


    def reset(self, l_rate = 0, beta = 0, batch = 0):
        # reset weight vector to zero, specify how to reset fields
        self.w = np.zeros(np.shape(self.x_train[0]))
        self.lr = l_rate
        self.beta = beta
        self.batch = batch


if __name__ == '__main__':
    optim = graddesc('pumadyn32nm', iter = 5, l_rate = 0.05)
    optim.run_gd()
    print(np.shape(optim.w))
    