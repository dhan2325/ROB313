import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
import time

def load_cyl2d_dataset(print_details=False):
    """
    Loads the cylinder flow dataset.

    Inputs:
        print_details : (bool, optional) whether to print details about the dataset

    Outputs:
        x_train, x_valid, x_test, y_train, y_valid, y_test
    """
    with np.load('cylinder2d.npz') as data:
        u = data['u']
        v = data['v']
        xdim = data['x']
        ydim = data['y']
        tdim = data['t']

        # for time series dataset, feature-target pairs are (time, state)
        # even though state at a given time is 160x80x2, we flatten it to a 25600-long vector
        x = tdim
        y = np.reshape(np.stack((u, v), axis=3), (tdim.shape[0], -1))

        # divide the dataset into 1000 for training, 200 for validation, 300 for testing
        x_train = x[:1001]
        y_train = y[:1001]

        x_valid = x[1001:1201]
        y_valid = y[1001:1201]

        x_test = x[1201:1501]
        y_test = y[1201:1501]

    if print_details:
        print("Cylinder Flow Dataset")

        print("xdim = %d" % xdim.shape)
        print("ydim = %d" % ydim.shape)
        print("tdim = %d" % tdim.shape)

        print("d = %d" % x_train.shape[1])
        print("n_train = %d" % x_train.shape[0])
        print("n_valid = %d" % x_valid.shape[0])
        print("n_test = %d" % x_test.shape[0])

    return x_train, x_valid, x_test, y_train, y_valid, y_test, xdim, ydim

def find_pca_matrix(y_train, z_d):
    """
    Finds the matrix that encodes/decodes a dataset via PCA.

    Inputs:
        y_train : dataset to be encoded
        z_d : dimension of encoded state

    Outputs:
        pca_matrix : the PCA mode matrix, and 
        pca_vector : the PCA mean vector, where np.matmul(y_train - pca_vec, pca_matrix) = z_train
    """
    # Compute the mean of the dataset
    pca_vec = np.mean(y_train, axis=0)

    # Center the dataset by subtracting the mean
    y_centered = y_train - pca_vec

    # Perform singular value decomposition (SVD)
    u, s, vh = np.linalg.svd(y_centered, full_matrices=False)

    # Take the first z_d columns of vh to form the PCA matrix
    pca_matrix = vh[:z_d, :].T

    return pca_matrix, pca_vec

def sqexp_kernel(x, z, theta=1, variance=1.):
    """ one-dimensional squared exponential kernel with lengthscale theta """
    # x = x.reshape((len(x), 1))
    # z = z.reshape((len(z), 1))
    return variance * np.exp(-np.square(x - z.T) / theta)

def matern_kernel(x, z, theta=1, variance=1.):
    """ one-dimensional matern kernel with lengthscale theta """
    """ YOUR CODE HERE """
    pass

def gp_prediction(x, y, x_test, kernel, noise_var = 1e-6):
    """ Computes posterior mean and cov at x_test given data x, y """
    """ THIS IS CURRENTLY FOR SCALAR TARGETS ONLY """

    x = x.reshape((-1, 1)) # each scalar entry should not be its own subarray
    y = y.reshape((-1, 1))
    x_test = x_test.reshape((-1, 1))

    # number of training, test points
    N = x.shape[0]
    N_test = x_test.shape[0]

    C = cho_factor(kernel(x, x) + noise_var*np.identity(N))

    mu_pred = kernel(x_test, x).dot(cho_solve(C, y))
    
    cov_pred = (
        kernel(x_test, x_test) + noise_var*np.identity(N_test)
        - kernel(x_test, x).dot(cho_solve(C, kernel(x, x_test)))
    )
    return mu_pred, cov_pred


def gp_pred_multidim(x : np.ndarray, y, x_test, kernel, noise_var = 1e-6):
    """
    we have four scalar targets (elements of y),
    each of which need to be tested across all time steps (time steps in x_test)
    perform iteratively using the same time steps every time, but a different column of y_test
    return stacked matrices of all four variables in one
    """
    print(y.shape)
    D = y.shape[1]
    N = x.shape[0]
    N_test = x_test.shape[0]
    x_test = x_test.reshape((-1, 1))
    list_mu = []
    list_cov = []
    for dim in range(D):
        # x, x_test are the same every time
        y_i = y[:,dim] # extract only the colummn of i we want
        y_i = y_i.reshape((-1,1))
        
        C = cho_factor(kernel(x, x) + noise_var*np.identity(N))
        
        mu_pred = kernel(x_test, x).dot(cho_solve(C, y_i))
        list_mu.append(mu_pred)
        
        cov_pred = (
            kernel(x_test, x_test) + noise_var*np.identity(N_test)
            - kernel(x_test, x.reshape((-1,1))).dot(cho_solve(C, kernel(x.reshape((-1,1)), x_test)))
        )
        list_cov.append(cov_pred)
    
    mu = np.hstack(list_mu)
    cov = np.stack(list_cov, axis = 2)
    print("shape of mean covariance matrices: {}, {}".format(mu.shape, cov.shape))
    return mu, cov
    


def gp_evidence(x, y, kernel, noise_var):
    """ Computes the GP log marginal likelihood """
    """ THIS IS CURRENTLY FOR SCALAR TARGETS ONLY """
    N = x.shape[0]

    C = cho_factor(kernel(x, x) + noise_var*np.identity(N))

    log_evidence = (
        0.5*y.T.dot(cho_solve(C, y))
        - np.sum(np.log(np.diag(C[0])))
        - 0.5*N*np.log(2*np.pi)
    )

    return log_evidence
    # will return a single scalar of the evidence for the scalar target

def gp_ev_multidim(x, y, kernel, noise_var):
    N = x.shape[0]
    C = cho_factor(kernel(x, x) + noise_var*np.identity(N))    
    D = x.shape[1]
    logs = []
    for dim in range(D):
        y_i = y[:,dim]
        y_i = y_i.reshape((-1,1))
        
        log_i = (
        0.5*y_i.T.dot(cho_solve(C,y))
        -np.sum(np.log(np.diag(C[0])))
        - 0.5*N*np.log(2*np.pi)
        )
        
        logs.append(log_i)
    log_evidence = np.vstack(log_evidence)
    return log_evidence
        
        
        
    

if __name__ == "__main__":
    # loading data
    x_train, x_valid, x_test, y_train, y_valid, y_test, xdim, ydim = load_cyl2d_dataset()
    # print("shape of y: {}, {}".format(y_train.shape[0], y_train.shape[1]))

    
    
    # state is currently flattened, keep track of true shape for plotting
    state_shape = [ydim.shape[0], xdim.shape[0], 2]

    # define pca dimension
    z_d = 4

    # plot state at last time step of training set
    f, axarr = plt.subplots(1, 2)
    """ img = axarr[0].imshow(np.reshape(y_train[-1], state_shape)[:,:,0])
    f.colorbar(img, ax=axarr[0])
    img = axarr[1].imshow(np.reshape(y_train[-1], state_shape)[:,:,1])
    f.colorbar(img, ax=axarr[1])
    plt.show() """

    # do pca
    print("finding PCA conversion...   ", end='')
    pca_matrix, pca_vec = find_pca_matrix(y_train, z_d) # determine from training set, use to predict on valid, test sets
    # pca_m_test, pca_v_test = find_pca_matrix(y_test, z_d)
    print('done')


    '''
    for Q2: find MSE between the original and reconstruction for the last vector
    '''
    y_f = y_test[-1]
    avg = np.mean(y_f)
    #y_f_reconst = ((pca_matrix.T).dot(pca_matrix)).dot(y_f - pca_vec) + pca_vec
    # print(pca_matrix.shape, pca_matrix.T.shape, pca_vec.shape)
    y_f_reconst = np.matmul(pca_matrix, np.matmul(pca_matrix.T, y_f - pca_vec)) + pca_vec
    print( np.mean( y_f-y_f_reconst)**2 )
    print('average value in y_f was', avg)

    # encode flow state
    print('determining latent states for flow...', end = '')
    z_train = np.matmul(y_train - pca_vec, pca_matrix)
    z_valid = np.matmul(y_valid - pca_vec, pca_matrix)
    z_test = np.matmul(y_test - pca_vec, pca_matrix)
    print('done')
    
    # plot the PCA converted targets
    print(z_train.shape)
    z1, z2, z3, z4 = [], [], [], []
    x = range(1501)
    for z_point in z_train:
        z1.append(z_point[0]); z2.append(z_point[1]); z3.append(z_point[2]); z4.append(z_point[3])
    for z_point in z_valid:
        z1.append(z_point[0]); z2.append(z_point[1]); z3.append(z_point[2]); z4.append(z_point[3])
    for z_point in z_test:
        z1.append(z_point[0]); z2.append(z_point[1]); z3.append(z_point[2]); z4.append(z_point[3])
    
    fig, axs = plt.subplots(2,2)
    axs[0,0].plot(x, z1); axs[0,0].set_title('z1')
    axs[0,1].plot(x, z2); axs[0,1].set_title('z2')
    axs[1,0].plot(x, z3); axs[1,0].set_title('z3')
    axs[1,1].plot(x, z4); axs[1,1].set_title('z4')

    '''
    For Q3: perform the multidimensional predictions for the latent states
    '''
    z_test_pred_mean, z_test_pred_cov = gp_pred_multidim(x_train, z_train, x_test, sqexp_kernel)
    
    '''
    for Q4: find the GP log marginal likelihood for all four latent states
    '''
    
    

    # do type-ii inference for kernel hyperparameters
    """ YOUR CODE HERE """

    # do gp prediction over validation and test sets
    """ YOUR CODE HERE """

    # decode predictions
    """ y_valid_pred_mu = np.matmul(z_valid_pred_mean, pca_matrix) + pca_vec
    y_test_pred_mu = np.matmul(z_test_pred_mean, pca_matrix) + pca_vec
 """
    # plot prediction at final time step
    """ YOUR CODE HERE """