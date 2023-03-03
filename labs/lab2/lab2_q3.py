from data.data_utils import load_dataset # include copy of data folder in submission zip
import numpy as np
from time import time
import math
from matplotlib import pyplot as plt
import scipy.linalg as lin

'''
construct a RBF model to minimize least-squares loss function, using a guassian kernel. Consider different values of theta and regularization parameters
use cholesky factorization for the model.
    - use validation set to determine hyperparamters (rosenbrock, n = 1000, d = 2 and mauna_loa)
    - then, predict on test set using both training and validation sets
    - present test RMSE
'''


# =================== kernel functions ===================
'''
the gaussian functions that define the kernel
for our purposes, we were only given scalar valuesa of theta, and thus only use isotropic version of the Gaussian kernel
the anisotropic version is provided for completeness
'''

def iso_gaussian(x : np.ndarray, z : np.ndarray, theta : float):
    # transpose z: column minus row means np will automatically expand dimensions into an NxM matrix
    # take element-wise difference, divide all terms by theta, and perform element-wise exponentiation to build gram matrix
    if np.shape(x)[1] == 1:
        return np.exp(-np.square(x-np.transpose(z))/theta)
    # for 2D inputs?
    else:
        x =  np.expand_dims(x, axis = 1) # x is 1000 x 2
        z = np.expand_dims(z, axis = 0) # x is 1000 x 2
        # print(np.shape(np.exp(-np.square(x -z)/theta)))
        return np.exp(-np.sum(np.square(x-z)/theta, axis = 2, keepdims = False))

def aniso_gaussian(x : np.ndarray, z : np.ndarray, theta_inv: np.ndarray):
    return math.exp(-np.matmul(np.transpose(x-z), np.matmul(theta_inv, (x-z))))


# ================= code for RBF ========================
class RBF:
    def __init__(self, dataset: str, theta : float, reg_factor : float, test = False):
        self.theta = theta
        self.reg =reg_factor
        if dataset == 'mauna_loa':
            self.x_train, self.x_valid, self.x_test, self.y_train, self.y_valid, self.y_test = load_dataset(dataset)
        elif dataset == 'rosenbrock':
            self.x_train, self.x_valid, self.x_test, self.y_train, self.y_valid, self.y_test = load_dataset(dataset, n_train = 1000, d = 2)
        else:
            raise Exception("please enter a valid dataset (mauna_loa or rosenbrock)")
        
        self.test = test
        self.inv_matrix, self.K = None, None
        # initialize fields arbitrarily, will be set later by setup functions

        if self.test:
            self.x_train, self.y_train = np.vstack([self.x_train, self.x_valid]), np.vstack([self.y_train, self.y_valid])
            self.K = iso_gaussian(self.x_train, self.x_train, self.theta) # gram matrix from training data
            self.C = lin.cho_factor(self.K + self.reg * np.identity(np.shape(self.x_train)[0])) # decomposition of (K + lambda * I)
            self.alpha = lin.cho_solve(self.C, self.y_train) # solve system of equations for alpha given y
            self.pred = iso_gaussian(self.x_test, self.x_train, self.theta).dot(self.alpha)
            # last line, we determine the gram matrix for the test input, and dot it with alpha to find the prediction
        else:
            self.K = iso_gaussian(self.x_train, self.x_train, self.theta)
            self.C = lin.cho_factor(self.K + self.reg * np.identity(np.shape(self.x_train)[0]))
            self.alpha = lin.cho_solve(self.C, self.y_train)
            self.pred = iso_gaussian(self.x_valid, self.x_train, self.theta).dot(self.alpha)

    def get_rmse(self):
        if not self.test:
            return np.linalg.norm(self.pred - self.y_valid) / math.sqrt(self.pred.size)
        else:
            return np.linalg.norm(self.pred - self.y_test) / math.sqrt(self.pred.size)
            

        
        


if __name__ == '__main__':
    thetas = [0.05, 0.1, 0.5, 1, 2]
    lambdas = [0.001, 0.01, 0.1, 1]
    dataset = 'rosenbrock'
    # looks like for both datasets, the smallest values of theta, lambda yield lowest values of RMSE
    start = time()

    """ rbf = RBF(dataset, 0.05, 1)
    f = open('q3_' + dataset + '.txt', 'w')
    for l in lambdas:
        for t in thetas:
            rbf= RBF(dataset, t, l, test = False)
            f.write('Theta = ' + str(t) + ', Lambda = ' + str(l) + ', RMSE = ' + str(round(rbf.get_rmse(), 4)) + '\n')
    f.write('validation across the hyperparamter grid for ' + dataset + ' took ' + str(round(time()-start, 1)) + 's')
    f.close() """

    rbf = RBF(dataset, 1, 0.001, test = True) # enter best hyperparams for each dataset
    print(rbf.get_rmse())
    
    