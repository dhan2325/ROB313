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


# =================== kernel functions ===================
'''
the gaussian functions that define the kernel
for our purposes, we were only given scalar valuesa of theta, and thus only use isotropic version of the Gaussian kernel
the anisotropic version is provided for completeness
'''

def iso_gaussian(x : np.ndarray, z : np.ndarray, theta : float):
    return math.exp(-(np.linalg.norm((x-z), 2) ** 2) / theta)

def aniso_gaussian(x : np.ndarray, z : np.ndarray, theta_inv: np.ndarray):
    return math.exp(-np.matmul(np.transpose(x-z), np.matmul(theta_inv, (x-z))))


# ================= code for RBF ========================
class RBF:
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
        self.set_hyperparams(theta, reg_factor)

    def run_validation(self):
        loss = 0
        alpha = self.find_alpha(self.y_train) # alpha is determined for all the desired outputs for a set of inputs
        for i in range(np.shape(self.x_valid)[0]): # for each point in validation set
            # for the given datasets, x data points can be 1 or 2 dimensional, but output is always a scalar
            x, y = self.x_train[i], self.y_train[i][0]
            

            f_hat = 0
            for j in range(self.N):
                # will take gaussian of the test point x with all N training points
                f_hat += alpha[j] * iso_gaussian(x, self.x_train[j], self.theta)
            
            loss += (f_hat - y) ** 2
        
        loss = math.sqrt(loss / (np.shape(self.x_valid)[0]))
        return loss

    def run_test(self):
        self.x_train, self.y_train = np.vstack([self.x_train, self.x_valid]), np.vstack([self.y_train, self.y_valid])
        # very similar procedure to validation, except now we are using the extended training set and measuring against
        # the training set and not the validation set


    def set_hyperparams(self, new_theta, new_lambda):
        self.theta = new_theta
        self.reg = new_lambda # lambda is a keyword in python goddammit

        # will have to compute the K and inv_matrix again
        self.find_K()
        self.find_inv_matrix()

    
    def find_alpha(self, y : np.ndarray):
        return np.matmul(self.inv_matrix, y)
    
    def find_K(self):
        self.N = np.shape(self.x_train)[0]
        self.K = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range (self.N):
                self.K[i][j] = iso_gaussian(self.x_train[i], self.x_train[j], self.theta)
    
    def find_inv_matrix(self):
        to_invert = self.K + self.reg * np.identity(self.N)
        L_inv : np.ndarray = np.linalg.inv(np.linalg.cholesky(to_invert))
        self.inv_matrix = np.matmul(np.conj(L_inv).transpose(), L_inv)
        
        


if __name__ == '__main__':
    thetas = [0.05, 0.1, 0.5, 1, 2]
    lambdas = [0.000001, 0.001, 0.01, 0.1]
    dataset = 'rosenbrock'
    # looks like for both datasets, the smallest values of theta, lambda yield lowest values of RMSE
    start = time()
    rbf = RBF(dataset, 0.05, 1)
    f = open('q3_' + dataset + '.txt', 'w')
    for t in thetas:
        for l in lambdas:
            rbf.set_hyperparams(new_theta = t, new_lambda = l)
            f.write('Theta = ' + str(t) + ', Lambda = ' + str(l) + ', RMSE = ' + str(round(rbf.run_validation(), 4)) + '\n')
    f.write('validation across the hyperparamter grid for ' + dataset + ' took ' + str(round(time()-start, 1)) + 's')
    f.close()
    