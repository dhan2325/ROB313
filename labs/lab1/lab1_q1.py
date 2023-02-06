'''
retrieve datasets, split into 5 partitions for cross-validation to check hyperparameters
in addition, determine which of the l1, l2 metrics yields better RMSE loss
for each dataset, compute nearest neightbors using 'brute force' - for each element in the dataset,
determine its distance from every other element in the dataset
perform analysis for every dataset
'''


from data.data_utils import load_dataset # include copy of data folder in submission zip
import numpy as np
from math import sqrt
from typing import Callable
from queue import PriorityQueue as pq

'''
x train, x valid, x test, y train, y valid, y test = load dataset('mauna loa')
x train, x valid, x test, y train, y valid, y test = load dataset('rosenbrock', n train=5000, d=2)
x train, x valid, x test, y train, y valid, y test = load dataset('pumadyn32nm')
x train, x valid, x test, y train, y valid, y test = load dataset('iris')
x train, x valid, x test, y train, y valid, y test = load dataset('mnist small')
'''

# l1 is manhattan dist, l2 is euclidian dist

def l1_vec(vec1 : np.ndarray, vec2 : np.ndarray):
    assert vec1.shape() == vec2.shape(), "cannot compute distance for vectors of different dimensions"
    distance = 0
    for i in range(vec1.shape[0]):
        distance += abs(vec1[i] - vec2[i])
    return distance

def l2_vec(vec1 : np.ndarray, vec2 : np.ndarray):
    assert vec1.shape() == vec2.shape(), "cannot compute distance for vectors of different dimensions"
    distance = 0
    for i in range(vec1.shape[0]):
        distance += (vec1[i] - vec2[i])**2
    return sqrt(distance)

def l1_vals(n1, n2):
    return abs(n1-n2)
def l2_vals(n1, n2):
    return sqrt((n1-n2)**2)


def randomize(array_x : np.ndarray, array_y: np.ndarray, split_axis : int = 1):
    assert array_x.shape == array_y.shape
    joined = np.concatenate((array_x, array_y), axis=split_axis)
    
    np.random.shuffle(joined)
    [shuf_x, shuf_y] = np.split(joined, 2, axis = 1)
    return shuf_x, shuf_y

def eval_knn(k_range, x_set:np.ndarray, y_set:np.ndarray, x_test: np.ndarray, y_test:np.ndarray, dist : Callable):
    # pass in k value, the x and y sets for model and for evaluation, and specify distance metric function
    sum_cost = 0
    neighbors = pq() # priority queue sorted by distance to each neighbor
    costs = [[]*(x_test.size[0])]*k_range # store cost for 
    '''
    take a single point from x_test, find its k nearest neighbors using a priority queue
    pq containts all points in x_set, y_set: compute distance from test point using x value(s),
    and then store in pq a tuple of (cost, y_value). Return predicted result for x_test point
    using average of the k nearest neighbors.
    '''
    for i in range(x_test.size[0]):
        testpoint : tuple = (x_test[i], y_test[i])
        for j in range(x_set.size[0]):
            pq.put((dist(x_test[i], x_set[j]), y_set[j]))
        
    return sum_cost
    


# to keep variable naming and code readable, perform regression for each dataset in individual functions
def knn_mauna():
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('mauna_loa') # output is np arrays
    # use single set for validations and training
    # and a separate test for testing (once hyperparameters have been chosen)
    x_train, y_train = np.vstack([x_train, x_valid]), np.vstack([y_train, y_valid])
    
    assert x_train.shape == y_train.shape, "training data shape invalid"
    

    # want even multiples of five, discarding an insignificant amount of data
    while(x_train.shape[0] %5):
        x_train = x_train[:-1]
    while(y_train.shape[0] %5):
        y_train = y_train[:-1]
    for i  in range (3):
        print(x_train[i], y_train[i])
    
    x_train, y_train = randomize(x_train, y_train, split_axis = 1)
    [xt1, xt2, xt3, xt4, xt5] = np.split(x_train, 5, axis = 0)
    [yt1, yt2, yt3, yt4, yt5] = np.split(y_train, 5, axis = 0)
    
    '''
    k_NN for all of the five values of k, determine which value of k might work best
    for each value of k, perform five cross-validations - which value works best?
    for determining which value works best: calculate sum of l1/l2 distances for a specific method
    for determining k_NN neighbors, we will use a 'brute force' method that does not consider
    use of more complex data structures.
    '''

    

    


def maunua_cross_val(x_data : list[np.ndarray], y_data : list[np.ndarray], dist : Callable, k : int):
    '''
    output the total costs for each value of k in array
    construct queues for each point in the current validation set
    '''
    # take rmse of the costs for all points in validation set, for each value of k
    k_costs : list[int]= []
    
    for a in range(len(x_data)):
        pq_list : list[pq] = []
        costs = [0,0,0,0,0]
        # separate validation set, training set
        x_val, y_val = x_data[a], y_data[a]
        x_train, y_train = np.vstack([x_data[:a], x_data[a+1:]]), np.vstack([y_data[:a] + y_data[a+1:]])
        for i in range(x_val.size[0]):
            nq = pq()
            x_val_i = x_val[i]
            y_val_i = y_val[i]

            for j in range(x_train.size[0]):
                # to a single pq, add all points in the training set compared to a single validation point
                distance = dist(x_val_i, x_train[j])
                nq.put((distance, y_train[i]))
            # using the pq, determine the error for this particular validation point
            
            neighbours = []
            for b in range(1,k+1):
                neighbours.append(nq.get())
                costs[i] += rmse(y_val_i,neighbours)

    

def rmse(true, predictions : list[tuple]):
    return 0





if __name__ == "__main__":
    knn_mauna()