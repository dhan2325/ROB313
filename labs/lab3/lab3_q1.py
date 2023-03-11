from data.data_utils import load_dataset # include copy of data folder in submission zip
import numpy as np
from time import time
import math
from matplotlib import pyplot as plt


np.random.seed = 1006842534

class graddesc:
    def __init__(self, dataset, l_rate = 0, beta = 0, batch = 0):
        self.x_train, self.x_valid, self.x_test, self.y_train, self.y_valid, self.y_test = load_dataset(dataset)
        np.random.shuffle(self.x_train), np.random.shuffle(self.y_train)
        self.x_train, self.y_train = self.x_train[:1000], self.y_train[:1000]

        self.lr = l_rate
        self.beta = beta
        self.batch = batch
        self.w = np.zeros(np.shape(self.x_train[0]))
    
    def df_dw(self):
        return - self.x_train.T.dot(self.y_train) - np.matmul(self.x_train.T, self.x_train).dot(self.w)

    def run_gd(self):
        if self.beta != 0:
            self.run_momentum()
        elif self.batch != 0:
            self.run_stoch()
        else:
            self.run_full()
    
    def run_momentum(self):
        pass

    def run_stoch(self):
        pass

    def run_full(self):
        pass

    def reset(self, l_rate = 0, beta = 0, batch = 0):
        self.w = np.zeros(np.shape(self.x_train[0]))
        self.lr = l_rate
        self.beta = beta
        self.batch = batch
