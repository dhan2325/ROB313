from data.data_utils import load_dataset # include copy of data folder in submission zip
import numpy as np
from math import sqrt
from typing import Callable
from queue import PriorityQueue as pq
from time import time

'''
x train, x valid, x test, y train, y valid, y test = load dataset('mauna loa')
x train, x valid, x test, y train, y valid, y test = load dataset('rosenbrock', n train=5000, d=2)
x train, x valid, x test, y train, y valid, y test = load dataset('pumadyn32nm')
x train, x valid, x test, y train, y valid, y test = load dataset('iris')
x train, x valid, x test, y train, y valid, y test = load dataset('mnist small')
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
    joined = np.concatenate((array_x, array_y), axis=split_axis)
    
    np.random.shuffle(joined)
    [shuf_x, shuf_y] = np.split(joined, [2], axis = 1)
    return shuf_x, shuf_y



def knn_puma(k_max : int, shuffle : bool = True):
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('pumadyn32nm')
    # use single set for validations and training
    # and a separate test for testing (once hyperparameters have been chosen)
    x_train : np.ndarray = np.vstack([x_train, x_valid])
    y_train: np.ndarray = np.vstack([y_train, y_valid])
    
    assert x_train.shape[0] == y_train.shape[0], "training data shape invalid"
    if shuffle:
        x_train, y_train = randomize(x_train, y_train)
    # print(x_train.shape, y_train.shape)

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
    k_costs = np.array(puma_cross_val([xt1, xt2, xt3, xt4, xt5], [yt1, yt2, yt3, yt4, yt5], l2_vec, k_max))
    k_costs.round(decimals = 3)
    return k_costs

    

    

# without randomization, one of the cross-val parititions has significantly higher cost
# could try randomizing ordering before partitioning
def puma_cross_val(x_data : 'list[np.ndarray]', y_data : 'list[np.ndarray]', dist : Callable, k : int):
    '''
    output the total costs for each value of k in array
    construct queues for each point in the current validation set
    '''
    # take rmse of the costs for all points in validation set, for each value of k
    k_costs = []
    
    for a in range(len(x_data)):
        # pq_list : list[pq] = [] # one pq_list for each of the validation points
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
        for i in range(x_val.shape[0]):
            nq = pq()
            x_val_i = x_val[i] # (2x1) ndarray
            y_val_i = y_val[i] # scalar

            for j in range(x_train.shape[0]):
                # to a single pq, add all points in the training set compared to a single validation point
                distance = dist(x_val_i, x_train[j])
                nq.put((distance, y_train[j]))
            short_nq = pq()
            for _ in range(k):
                short_nq.put(nq.get())

            # pq_list.append(short_nq) # only store the k nearest neighbors
            # get the errors for each value of k
            y_pred = np.array([0] * y_train.shape[1])
            for h in range(1,k+1):
                new_y = short_nq.get(block = False)
                y_pred = y_pred *(h-1)/h + (new_y[1]) / h
                #print(y_pred.shape)
                costs[h-1] += (np.absolute(y_val_i - y_pred)).mean(axis = 0) # mean absolute error for each coord for a single point
        for i in range(k):
            costs[i] = sqrt(costs[i] / (y_val.shape[0])) # divide by number of points, take sqrt
        k_costs.append(costs)
        print('completed a partition')
    return k_costs





if __name__ == "__main__":
    f = open('puma_knn_costs.txt', 'w')
    k_max = 50
    start = time()
    costs = knn_puma(k_max, shuffle = False)
    dur = time() - start
    f.write("Duration (s): " + str(dur) + '\n')
    f.write("Max k value tested: " +str(k_max) + '\n\n')
    f.write(np.array2string(costs))

    