from data.data_utils import load_dataset # include copy of data folder in submission zip
import numpy as np
from time import time
import math
from matplotlib import pyplot as plt
'''
same graddesc class as before, we just have to change the df_dw methods
since we are minimizing a different function
'''
def sig(z):
    return 1/(1+np.exp(-z))

def randomize(array_x : np.ndarray, array_y: np.ndarray, split_axis : int = 1):
    joined = np.hstack([array_x, array_y])
    # print(np.shape(joined))
    
    np.random.shuffle(joined)
    [shuf_x, shuf_y] = np.split(joined, [-1], axis = 1)
    # print(np.shape(shuf_x), np.shape(shuf_y))
    return shuf_x, shuf_y

class graddesc:
    # gradient descent class, able to perform multiple variations on same dataset
    # weight vector still have one more element than a single x input
    def __init__(self, dataset, iter = 100, l_rate = 0, beta = 0, batch = 0, thresh = 0.1):
        self.x_train, self.x_valid, self.x_test, self.y_train, self.y_valid, self.y_test = load_dataset(dataset)
        self.x_train, self.y_train = randomize(self.x_train, self.y_train)
        self.x_train = np.hstack([np.ones((np.shape(self.x_train)[0], 1)), self.x_train])
        self.y_train = self.y_train[:,1]
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
        self.comp_time = 0

    def get_loss(self):
        # get squared loss using current weight vector
        return np.sum(np.square(self.y_train - self.x_train.dot(self.w)))
    def f_hat(self, x):
        return sig(self.w.T.dot(x))

    def df_dw(self):
        # determine gradient for current value of w
        # print(np.shape(self.x_train.T.dot(self.x_train.dot(self.w))))
        grad = np.zeros((np.shape(self.x_train)[0], 1))
        for i in range(np.shape(self.x_train)[0]):
            addgrad = (self.y_train[i] - self.f_hat(self.x_train[i])) * self.x_train[i]
            grad += addgrad
        return grad
    
    def df_dw_batch(self, batch_ind):
        # determine gradient for current value of w
        # print(np.shape(self.x_train.T.dot(self.x_train.dot(self.w))))
        x = self.x_train[batch_ind:batch_ind + self.batch]
        y = self.y_train[batch_ind:batch_ind + self.batch]
        # print(np.shape(x), np.shape(y))
        w = self.w
        grad = 2 * x.T.dot(x.dot(w) - y)
        # grad is 
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
        print('w momentum')
        # run gradient descent with momentum, using the Beta and batch size specified upon declaration
        cur_batch = 0
        done = False
        prev_grad = self.df_dw_batch(cur_batch) # first gradient approximation is just first gradient
        iter_counter = 0
        for i in range(self.iter):
            start = time()
            grad = self.beta * prev_grad + (1 - self.beta) * self.df_dw_batch(cur_batch)
            self.w = self.w - self.lr * grad
            cur_batch = (cur_batch + self.batch) % 1000 # index for next batch increased, wrap
            self.comp_time += time()-start
            newloss = self.get_loss()
            self.losses.append(newloss)
            prev_grad = grad
            if (not done) and ((newloss  - self.ls_loss)/self.ls_loss < self.thresh / 100):
                done = True
                print('{}% of loss at {} iterations'.format(self.thresh, i))
            if iter_counter == 1000 / self.batch:
                iter_counter = 0
                self.losses_epoch.append(newloss)
            else:
                iter_counter += 1
        if not done:
            print('did not reach {}% of loss after {} iterations'.format(self.thresh, self.iter))
        return self.w

    def run_stoch(self):
        # run sotchastic gradient descent, using the batch size specified upon declaration
        cur_batch = 0
        done = False
        iter_counter = 0
        for i in range(self.iter):
            start = time()
            diff = self.lr * self.df_dw_batch(cur_batch)
            self.w = self.w - diff
            cur_batch = (cur_batch + self.batch) % 1000 # index for next batch increased
            self.comp_time+= time()-start
            newloss = self.get_loss()
            self.losses.append(newloss)
            if (not done) and ((newloss  - self.ls_loss)/self.ls_loss < self.thresh/100):
                done = True
                print('{}% of loss at {} iterations'.format(self.thresh, i))
            if iter_counter == 1000 / self.batch:
                iter_counter = 0
                self.losses_epoch.append(newloss)
            else:
                iter_counter += 1
        if not done:
            print('did not reach {}% of loss after {} iterations'.format(self.thresh, self.iter))
        return self.w


    def run_full(self):
        # run a full gradient descent using all 1000 training points
        
        done = False
        for i in range(self.iter):
            start =time()
            #print(self.w[:3])
            diff = (self.lr * self.df_dw())
            self.w = self.w - diff
            #print(self.w[:3], '\n')
            self.comp_time += time()-start
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
    it = 100 # 100 * 1000/batch_size
    optim = graddesc('iris', iter = it, l_rate = 0.0001, batch = 10, beta = 0.9, thresh = 0.01)