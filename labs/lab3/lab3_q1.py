from data.data_utils import load_dataset # include copy of data folder in submission zip
import numpy as np
from time import time
import math
from matplotlib import pyplot as plt


np.random.seed = 1006842534

class graddesc:
    def __init__(self, dataset, iter = 100, l_rate = 0, beta = 0, batch = 0):
        self.x_train, self.x_valid, self.x_test, self.y_train, self.y_valid, self.y_test = load_dataset(dataset)
        np.random.shuffle(self.x_train), np.random.shuffle(self.y_train)
        self.x_train, self.y_train = self.x_train[:1000], self.y_train[:1000]
        print(np.shape(self.x_train.T.dot(self.y_train)))
        self.iter = iter
        self.lr = l_rate
        self.beta = beta
        self.batch = batch
        self.w = np.zeros((np.shape(self.x_train[0])[0], 1))

    
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
        for _ in range(self.iter):
            self.w = self.w  - self.lr * self.df_dw()


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
    