from data.data_utils import load_dataset # include copy of data folder in submission zip
import numpy as np
from time import time
import math
from matplotlib import pyplot as plt


np.random.seed = 1006842534

def randomize(array_x : np.ndarray, array_y: np.ndarray, split_axis : int = 1):

    joined = np.hstack([array_x, array_y])
    print(np.shape(joined))
    
    np.random.shuffle(joined)
    [shuf_x, shuf_y] = np.split(joined, [-1], axis = 1)
    print(np.shape(shuf_x), np.shape(shuf_y))
    return shuf_x, shuf_y

class graddesc:
    '''
    stopping condition for gradient descent? Currently implementing a 'relative error' decrease

    '''
    def __init__(self, dataset, iter = 100, l_rate = 0, beta = 0, batch = 0):
        self.x_train, self.x_valid, self.x_test, self.y_train, self.y_valid, self.y_test = load_dataset(dataset)
        self.x_train, self.y_train = self.x_train[1000:2000], self.y_train[1000:2000]
        # self.x_train, self.y_train = randomize(self.x_train, self.y_train)
        self.x_train = np.hstack([np.ones((np.shape(self.x_train)[0], 1)), self.x_train])
        # print(self.x_train[69])
        self.iter = iter
        self.lr = l_rate
        self.beta = beta
        self.batch = batch
        self.w = np.zeros((np.shape(self.x_train[0])[0], 1))
        print(np.shape(self.w))

        self.losses : list[int] = []

    
    def df_dw(self):
        # determine gradient for current value of w
        # print(np.shape(self.x_train.T.dot(self.x_train.dot(self.w))))
        x = self.x_train
        y = self.y_train
        w = self.w
        grad = 2 * x.T.dot(x.dot(w) - y)
        return grad
    
    def df_dw_batch(self, batch_ind):
        # determine gradient for current value of w
        # print(np.shape(self.x_train.T.dot(self.x_train.dot(self.w))))
        x = self.x_train[batch_ind:batch_ind + self.batch]
        y = self.y_train[batch_ind:batch_ind + self.batch]
        print(np.shape(x), np.shape(y))
        w = self.w
        grad = 2 * x.T.dot(x.dot(w) - y)
        return grad

    def run_gd(self):
        # run the appropriate variation of grad desc
        if self.beta != 0:
            return self.run_momentum()
        elif self.batch != 0:
            return self.run_stoch()
        else:
            return self.run_full()
    
    def run_momentum(self):
        # run gradient descent with momentum, using the Beta and batch size specified upon declaration
        pass

    def run_stoch(self):
        # run sotchastic gradient descent, using the batch size specified upon declaration
        cur_batch = 0
        for _ in range(self.iter):
            diff = self.lr * self.df_dw_batch(cur_batch)
            self.w = self.w - diff
            cur_batch += self.batch
        return self.w


    def run_full(self):
        # run a full gradient descent using all 1000 training points
        for _ in range(self.iter):
            #print(self.w[:3])
            diff = (self.lr * self.df_dw())
            self.w = self.w - diff
            #print(self.w[:3], '\n')
        return self.w


    def reset(self, l_rate = 0, beta = 0, batch = 0):
        # reset weight vector to zero, specify how to reset fields
        self.w = np.zeros((np.shape(self.x_train[0])[0], 1))
        self.lr = l_rate
        self.beta = beta
        self.batch = batch


if __name__ == '__main__':
    # breaking point: l_rate = 0.0008
    optim = graddesc('pumadyn32nm', iter = 3, l_rate = 0.0001, batch = 1)
    weight = optim.run_gd()
    
    truew, resid, rank, s = np.linalg.lstsq(optim.x_train.T.dot(optim.x_train), optim.x_train.T.dot(optim.y_train))
    y_preds = np.matmul(optim.x_train, truew)
    print(truew  - optim.w)

    # plt.plot(list(range(10)), (optim.w - truew)[:10], marker = 'o',markersize = 1, color = (0,0,1), linestyle = None, linewidth = 0)
    plt.plot(range(optim.y_train.size), y_preds - optim.y_train, marker = 'o', markersize = 1, color = (1,0,0), linestyle = None, linewidth = 0)
    plt.show()

    