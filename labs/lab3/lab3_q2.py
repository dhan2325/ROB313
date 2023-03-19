from data.data_utils import load_dataset # include copy of data folder in submission zip
import numpy as np
from time import time
import math
from matplotlib import pyplot as plt
from scipy.special import expit as sigmoid

def sig(z):
    # print(1/(1+np.exp(-z)))
    return 1/(1+np.exp(-z))

def randomize(array_x : np.ndarray, array_y: np.ndarray, split_axis : int = 1):
    joined = np.hstack([array_x, array_y])
    # print(np.shape(joined))
    
    np.random.shuffle(joined)
    [shuf_x, shuf_y] = np.split(joined, [-1], axis = 1)
    # print(np.shape(shuf_x), np.shape(shuf_y))
    return shuf_x, shuf_y

class graddesc:
    '''
    same graddesc class as before, we just have to change the df_dw methods
    since we are minimizing a different function
    '''
    # gradient descent class, able to perform multiple variations on same dataset
    # weight vector still have one more element than a single x input
    def __init__(self, dataset, iter = 100, l_rate = 0, beta = 0, batch = 0, thresh = 0.1):
        self.x_train, self.x_valid, self.x_test, self.y_train, self.y_valid, self.y_test = load_dataset(dataset)
        self.x_train, self.y_train = np.vstack([self.x_train, self.x_valid]), np.vstack([self.y_train, self.y_valid])
        self.x_train = np.hstack([np.ones((np.shape(self.x_train)[0], 1)), self.x_train])
        self.x_test = np.hstack([np.ones((np.shape(self.x_test)[0], 1)), self.x_test])
        self.y_train = np.reshape(self.y_train[:,2], (np.shape(self.x_train)[0], 1)) # extract single column
        self.y_test = np.reshape(self.y_test[:,2], (np.shape(self.x_test)[0], 1))
        self.y_train, self.y_test = self.y_train.astype(int), self.y_test.astype(int)
        print(np.shape(self.x_train), np.shape(self.y_train))
        # self.y_train = self.y_train[:,1]
        # print(self.x_train[69])
        self.iter = iter
        self.lr = l_rate
        self.beta = beta
        self.batch = batch
        self.w = np.zeros((np.shape(self.x_train[0])[0], 1))
        # print(np.shape(self.w))

        self.thresh = thresh
        self.losses : list[float] = []
        self.losses_epoch : list[float] = []
        self.losses_time : list[float] = []
        self.comp_time = 0

        self.bestw, self.bestloss = np.zeros((np.shape(self.x_train[0])[0], 1)), 100

    def get_loss(self):
        # get squared loss using current weight vector
        return np.sum(np.square(self.y_train - self.x_train.dot(self.w)))


    def df_dw(self):
        # determine gradient for current value of w
        # print(np.shape(self.x_train.T.dot(self.x_train.dot(self.w))))
        pred = self.x_train.dot(self.w)
        f_hat = 1/(1+np.exp(-pred))
        return -np.sum((self.y_train-f_hat)* self.x_train)
    
    def df_dw_batch(self, batch_ind):
        # determine gradient for current value of w
        # print(np.shape(self.x_train.T.dot(self.x_train.dot(self.w))))
        x = self.x_train[batch_ind:batch_ind + self.batch]
        y = self.y_train[batch_ind:batch_ind + self.batch]
        # print(np.shape(x), np.shape(y))
        pred = x.dot(self.w)
        f_hat = 1/(1+np.exp(-pred))
        return -np.sum((y-f_hat)*x)


    def run_gd(self):
        # run the appropriate variation of grad desc
        if self.beta != 0:
            return self.run_momentum()
        elif self.batch != 0:
            return self.run_stoch()
        else:
            print('performing full Gradient Descent')
            return self.run_full()
    


    def run_stoch(self):
        print('stochastic')
        # run sotchastic gradient descent, using the batch size specified upon declaration
        cur_batch = 0
        iter_counter = 0
        bl = 100
        for i in range(self.iter):
            start = time()
            diff = self.lr * self.df_dw_batch(cur_batch)
            self.w = self.w - diff
            cur_batch = (cur_batch + self.batch) % 1000 # index for next batch increased
            self.comp_time+= time()-start
            newloss = -self.log_likelihood()
            bl = min(newloss, bl)
            self.losses.append(bl)

            if iter_counter == 1000 / self.batch:
                iter_counter = 0
                if newloss < self.bestloss:
                    self.bestloss = newloss
                    self.bestw = self.w
                self.losses_epoch.append(min(newloss, self.bestloss))
            else:
                iter_counter += 1
        return self.w

    
    def run_full(self):
        # run a full gradient descent using all 1000 training points
        for i in range(self.iter):
            start =time()
            #print(self.w[:3])
            diff = (self.lr * self.df_dw())
            self.w = self.w - diff
            #print(self.w[:3], '\n')
            self.comp_time += time()-start
            newloss = -self.log_likelihood()
            if newloss < self.bestloss:
                self.bestloss = newloss
                self.bestw = self.w
            self.losses.append(min(newloss, self.bestloss))
        return self.w


    def get_accuracy(self):
        f_hat = sigmoid(self.x_test.dot(self.bestw))
        return np.mean((f_hat >0.5) == self.y_test)

    def log_likelihood(self):
        f_hat = sigmoid(self.x_train.dot(self.w))
        return np.sum(self.y_train * np.log(f_hat) + (1-self.y_train)*np.log(1-f_hat))

    def reset(self, l_rate = 0, beta = 0, batch = 0):
        # reset weight vector to zero, specify how to reset fields
        self.w = np.zeros((np.shape(self.x_train[0])[0], 1))
        self.lr = l_rate
        self.beta = beta
        self.batch = batch
        self.losses : list[float] = []
        self.losses_epoch : list[float] = []
        self.losses_time : list[float] = []
    
    def log_likelihood_test(self):
        f_hat = sigmoid(self.x_test.dot(self.w))
        return np.sum(self.y_test * np.log(f_hat) + (1-self.y_test)*np.log(1-f_hat))
    
    
if __name__ == '__main__':
    '''
    plot all GD, SGD on a single plot across epochs
    '''
    # breaking point: l_rate = 0.0008
    it = 25 # 100 * 1000/batch_size
    optim = graddesc('iris', iter = it, l_rate = 0.0001)
    optim.run_gd()
    for a in (0.00005, 0.0001, 0.0005, 0.001):
        optim.reset(l_rate = a)
        optim.run_gd()
        plt.plot(range(it), optim.losses, markersize = 1, label='GD, lr = {}'.format(a))
        print(optim.get_accuracy())
        print(optim.log_likelihood_test())
    plt.legend()
    plt.savefig('./labs/lab3/images/q2_full.png')


    it = it * 1000
    optim = graddesc('iris', iter = it, l_rate = 0.0001, batch = 1)
    plt.clf()
    for a in  (0.00005, 0.0001, 0.0005, 0.001):
        optim.reset(l_rate = a, batch = 1)
        optim.run_gd()
        plt.plot(range(int(it/1000 - 1)), optim.losses_epoch, markersize = 1, label='SGD lr = {}'.format(a))
        print(optim.get_accuracy())
        print(optim.log_likelihood_test())
    plt.legend()
    plt.savefig('./labs/lab3/images/q2_SGD.png')