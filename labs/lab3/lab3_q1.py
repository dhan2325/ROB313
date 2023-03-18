from data.data_utils import load_dataset # include copy of data folder in submission zip
import numpy as np
from time import time
import math
from matplotlib import pyplot as plt

# TODO: test for different values of learning rate, beta
# TODO: graph results

np.random.seed = 1006842534

def randomize(array_x : np.ndarray, array_y: np.ndarray, split_axis : int = 1):

    joined = np.hstack([array_x, array_y])
    # print(np.shape(joined))
    
    np.random.shuffle(joined)
    [shuf_x, shuf_y] = np.split(joined, [-1], axis = 1)
    # print(np.shape(shuf_x), np.shape(shuf_y))
    return shuf_x, shuf_y

class graddesc:
    # gradient descent class, able to perform multiple variations on same dataset
    def __init__(self, dataset, iter = 100, l_rate = 0, beta = 0, batch = 0, thresh = 0.1):
        self.x_train, self.x_valid, self.x_test, self.y_train, self.y_valid, self.y_test = load_dataset(dataset)
        self.x_train, self.y_train = randomize(self.x_train, self.y_train)
        self.x_train, self.y_train = self.x_train[:1000], self.y_train[:1000]
        self.x_train = np.hstack([np.ones((np.shape(self.x_train)[0], 1)), self.x_train])
        # print(self.x_train[69])
        self.iter = iter
        self.lr = l_rate
        self.beta = beta
        self.batch = batch
        self.w = np.zeros((np.shape(self.x_train[0])[0], 1))
        # print(np.shape(self.w))

        self.ls_loss, self.truew = self.get_ls_loss()
        self.thresh = thresh
        self.losses : list[float] = []
        self.losses_epoch : list[float] = []
        self.losses_time : list[float] = []

    def get_loss(self):
        # get squared loss using current weight vector
        return np.sum(np.square(self.y_train - self.x_train.dot(self.w)))
    
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
        # print(np.shape(x), np.shape(y))
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
            print('performing full Gradient Descent')
            return self.run_full()
    
    def run_momentum(self):
        # run gradient descent with momentum, using the Beta and batch size specified upon declaration
        cur_batch = 0
        done = False
        prev_grad = self.df_dw_batch(cur_batch) # first gradient approximation is just first gradient
        for i in range(self.iter):
            grad = self.beta * prev_grad + (1 - self.beta) * self.df_dw_batch(cur_batch)
            self.w = self.w - self.lr * grad
            cur_batch = (cur_batch + self.batch) % 1000 # index for next batch increased, wrap
            newloss = self.get_loss()
            self.losses.append(newloss)
            prev_grad = grad
            if (not done) and ((newloss  - self.ls_loss)/self.ls_loss < self.thresh / 100):
                done = True
                print('{}% of loss at {} iterations'.format(self.thresh, i))
        if not done:
            print('did not reach {}% of loss after {} iterations'.format(self.thresh, self.iter))
        return self.w

    def run_stoch(self):
        # run sotchastic gradient descent, using the batch size specified upon declaration
        cur_batch = 0
        done = False
        for i in range(self.iter):
            diff = self.lr * self.df_dw_batch(cur_batch)
            self.w = self.w - diff
            cur_batch = (cur_batch + self.batch) % 1000 # index for next batch increased
            newloss = self.get_loss()
            self.losses.append(newloss)
            if (not done) and ((newloss  - self.ls_loss)/self.ls_loss < self.thresh/100):
                done = True
                print('{}% of loss at {} iterations'.format(self.thresh, i))
        if not done:
            print('did not reach {}% of loss after {} iterations'.format(self.thresh, self.iter))
        return self.w


    def run_full(self):
        # run a full gradient descent using all 1000 training points
        
        done = False
        for i in range(self.iter):
            #print(self.w[:3])
            diff = (self.lr * self.df_dw())
            self.w = self.w - diff
            #print(self.w[:3], '\n')
            newloss = self.get_loss()
            self.losses.append(newloss)
            if (not done) and ((newloss  - self.ls_loss)/self.ls_loss < self.thresh/100):
                done = True
                print('{}% of loss at {} iterations with learning rate {}'.format(self.thresh, i, self.lr))
        if not done:
            print('did not reach {}% of loss after {} iterations with learning rate {}'.format(self.thresh, self.iter, self.lr))
        return self.w


    def reset(self, l_rate = 0, beta = 0, batch = 0):
        # reset weight vector to zero, specify how to reset fields
        self.w = np.zeros((np.shape(self.x_train[0])[0], 1))
        self.lr = l_rate
        self.beta = beta
        self.batch = batch
        self.losses : list[float] = []
        self.losses_epoch : list[float] = []
        self.losses_time : list[float] = []
    
    def get_ls_loss(self):
        truew, resid, rank, s = np.linalg.lstsq(self.x_train.T.dot(self.x_train), self.x_train.T.dot(self.y_train), rcond = None)
        y_preds = np.matmul(self.x_train, truew)
        ls_loss = np.sum(np.square(self.y_train - y_preds))
        return ls_loss, truew



if __name__ == '__main__':
    # breaking point: l_rate = 0.0008
    it = 100
    optim = graddesc('pumadyn32nm', iter = it, l_rate = 0.0001, batch = 1, beta = 0, thresh = 0.01)
    # weight = optim.run_gd()
    
    """ plt.plot(range(it), [optim.ls_loss] * it, markersize = 1, color = (0,0,1))
    plt.plot(range(it), optim.losses, markersize = 1, color = (1,0,0))
    print(optim.losses)
    plt.show() """

    a_range = (0.00005, 0.0001, 0.00025, 0.0005, 0.001)

    for a in a_range:
        optim.reset(l_rate = a, beta = 1)
        # print(optim.w)
        optim.run_gd()
        # print(optim.losses)
        # print(optim.losses)
        plt.clf()
        plt.title('Gradient Descent: learning rate = {}'.format(a))
        plt.plot(range(it), [optim.ls_loss] * it, markersize = 1, color = (0,0,1))
        plt.plot(range(it), optim.losses, markersize = 1, color = (1,0,0))
        plt.show()


    