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
    write function to determine K matrix once for the validation set, 
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
        
        self.K = None
        self.find_K()
        for i in range(3):
            print(self.x_train[i])

    def run_algo(self):
        pass
    
    def find_alpha(self):
        pass
    
    def find_K(self):
        N = np.shape(self.x_train)[0]
        print(type(N))
        self.K = np.zeros((N, N))
        for i in range(N):
            for j in range (N):
                self.K[i][j] = iso_gaussian(self.x_train[i], self.x_train[j], self.theta)
        


if __name__ == '__main__':
    rbf = RBF('rosenbrock', 0.05, 0)
    print(rbf.K)