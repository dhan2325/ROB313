from data.data_utils import load_dataset # include copy of data folder in submission zip
import numpy as np
from time import time
import math

'''
construct a RBF model to minimize least-squares loss function, using a guassian kernel. Consider different values of theta and regularization parameters
use cholesky factorization for the model.
    - use validation set to determine hyperparamters (rosenbrock, n = 1000, d = 2 and mauna_loa)
    - then, predict on test set using both training and validation sets
    - present test RMSE
'''


def iso_gaussian(x : np.ndarray, z : np.ndarray, theta : float):
    return math.exp(-(np.linalg.norm((x-z), 2) ** 2) / theta)

def aniso_gaussian(x : np.ndarray, z : np.ndarray, theta_inv: np.ndarray):
    return math.exp(-np.matmul(np.transpose(x-z), np.matmul(theta_inv, (x-z))))


class RBF:
    '''
    matrix K is independent of the input, and is formed from the kernels of the training points
    however, the single kernel term in the cost function must be computes for each test point independently
    write function to determine K matrix once for the validation set, which will be run once upon init
    since lambda, K are fixed, evaluate the inverse (using cholesky factorization) once, store in class
    write function to find alpha for a given point y, using the predetermined inverted matrix
    '''
    def __init__(self, dataset: str, theta : float, reg_factor : float):
        self.theta = theta
        self.reg =reg_factor
        if dataset == 'mauna_loa':
            self.x_train, self.x_valid, self.x_test, self.y_train, self.y_valid, self.y_test = load_dataset(dataset)
        elif dataset == 'rosenbrock':
            self.x_train, self.x_valid, self.x_test, self.y_train, self.y_valid, self.y_test = load_dataset(dataset, n_train = 1000, d = 2)
        else:
            raise Exception("please enter a valid dataset (mauna_loa or rosenbrock)")
        
        # initialize fields arbitrarily, will be set later by setup functions
        self.K = None
        self.inv_matrix = None
        self.N = 0

        # setup functions
        self.find_K()
        self.find_inv_matrix()

    def run_algo(self):
        pass
    
    def find_alpha(self, y : np.ndarray):
        return np.matmul(self.inv_matrix, y)
    
    def find_K(self):
        self.N = np.shape(self.x_train)[0]
        print(type(self.N))
        self.K = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range (self.N):
                self.K[i][j] = iso_gaussian(self.x_train[i], self.x_train[j], self.theta)
    
    def find_inv_matrix(self):
        to_invert = self.K + self.reg * np.identity(self.N)
        L_inv : np.ndarray = np.linalg.inv(np.linalg.cholesky(to_invert))
        self.inv_matrix = np.matmul(L_inv.H, L_inv)

        
        


if __name__ == '__main__':
    rbf = RBF('rosenbrock', 0.05, 0)
    print(rbf.K)