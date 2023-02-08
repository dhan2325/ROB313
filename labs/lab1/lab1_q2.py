from data.data_utils import load_dataset # include copy of data folder in submission zip
import numpy as np
from math import sqrt
from typing import Callable
from queue import PriorityQueue as pq
from time import time
from sklearn.neighbors import KDTree as kdt

'''
perform k_NN regression on the rosenbrock dataset, this time using a k-dimensional tree to store and query for neighbors
two versions implemented: the priority queue 'brute force' method and a k-dimensional tree, both for k = 5
runtimes compared using the l2 distance metric
'''

# l1 is manhattan dist, l2 is euclidian dist

def l1_vec(vec1 : np.ndarray, vec2 : np.ndarray):
    assert vec1.shape == vec2.shape, "cannot compute distance for vectors of different dimensions"
    return np.linalg.norm(vec1-vec2, ord = 1)
    
def l2_vec(vec1 : np.ndarray, vec2 : np.ndarray):
    assert vec1.shape == vec2.shape, "cannot compute distance for vectors of different dimensions"
    return np.linalg.norm(vec1-vec2, ord = 2)

def l1_vals(n1, n2):
    return abs(n1-n2)

def l2_vals(n1, n2):
    return sqrt((n1-n2)**2)

def randomize(array_x : np.ndarray, array_y: np.ndarray, split_axis : int = 1):
    assert array_x.shape[0] == array_y.shape[0]
    print(array_x.shape, array_y.shape)
    joined = np.concatenate((array_x, array_y), axis=split_axis)
    
    np.random.shuffle(joined)
    [shuf_x, shuf_y] = np.split(joined, [2], axis = 1)
    #print(shuf_x.shape, shuf_y.shape)
    return shuf_x, shuf_y



def knn_rosenbrock_kdt(k_max):
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('rosenbrock', n_train=1000, d=2)
    # use single set for validations and training
    # and a separate test for testing (once hyperparameters have been chosen)
    x_train, y_train = np.vstack([x_train, x_valid]), np.vstack([y_train, y_valid])
    
    assert x_train.shape[0] == y_train.shape[0], "training data shape invalid"
    x_train, y_train = randomize(x_train, y_train)

    # want even multiples of five, discarding an insignificant amount of data
    while(x_train.shape[0] %5):
        x_train = x_train[:-1]
    while(y_train.shape[0] %5):
        y_train = y_train[:-1]
    
    [xt1, xt2, xt3, xt4, xt5] = np.split(x_train, 5, axis = 0)
    [yt1, yt2, yt3, yt4, yt5] = np.split(y_train, 5, axis = 0)
    
    '''
    k_NN for all of the five values of k, determine which value of k might work best
    for each value of k, perform five cross-validations - which value works best?
    for determining which value works best: calculate sum of l1/l2 distances for a specific method
    for determining k_NN neighbors, we will use a 'brute force' method that does not consider
    use of more complex data structures.
    '''
    k_costs = np.array(rosenbrock_cross_val_kdt([xt1, xt2, xt3, xt4, xt5], [yt1, yt2, yt3, yt4, yt5], l2_vec, k_max))
    k_costs.round(decimals = 3)
    return k_costs

    

# without randomization, one of the cross-val parititions has significantly higher cost
# could try randomizing ordering before partitioning
def rosenbrock_cross_val_kdt(x_data : 'list[np.ndarray]', y_data : 'list[np.ndarray]', dist : Callable, k : int):
    '''
    output the total costs for each value of k in array
    construct queues for each point in the current validation set
    '''
    # take rmse of the costs for all points in validation set, for each value of k
    k_costs = []
    
    for a in range(len(x_data)):
        pq_list : list[pq] = [] # one pq_list for each of the validation points
        costs = [0] * k
        # separate validation set, training set
        x_val, y_val = x_data[a], y_data[a]
        # print(x_val.shape)
        if (a == 0):
            b = 1
        else:
            b = 0
        x_train, y_train = x_data[b], y_data[b]
        for i in range(len(x_data)):
            if (i!=a) and (i != b):
                x_train, y_train = np.vstack([x_train, x_data[i]]), np.vstack([y_train, y_data[i]])
        # x_train, y_train = np.vstack([x_data[:a], x_data[a+1:]]), np.vstack([y_data[:a] + y_data[a+1:]])
        
    return k_costs





if __name__ == "__main__":
    f = open('rb_knn_costs.txt', 'w')
    costs = knn_rosenbrock(7)
    for _ in range(10):
        costs += knn_rosenbrock(7)
    f.write(np.array2string(costs))
    