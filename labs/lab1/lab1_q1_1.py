from data.data_utils import load_dataset # include copy of data folder in submission zip
import numpy as np
from math import sqrt
from typing import Callable
from queue import PriorityQueue as pq
from time import time
import matplotlib.pyplot as plt

np.random.seed = 2325

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
    assert array_x.shape == array_y.shape
    joined = np.concatenate((array_x, array_y), axis=split_axis)
    
    np.random.shuffle(joined)
    [shuf_x, shuf_y] = np.split(joined, 2, axis = 1)
    return shuf_x, shuf_y

def maunua_cross_val(x_data : 'list[np.ndarray]', y_data : 'list[np.ndarray]', dist : Callable, k : int):
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
            x_val_i = x_val[i]
            y_val_i = y_val[i]
    

# to keep variable naming and code readable, perform regression for each dataset in individual functions
def knn_mauna(k_max: int, shuffle : bool = True):
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

    
    # option to randomize, set to true by default
    if shuffle:
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
    k_costs = maunua_cross_val([xt1, xt2, xt3, xt4, xt5], [yt1, yt2, yt3, yt4, yt5], l2_vals, k_max)
    """ for row in k_costs:
        print(row, '\n') """
    y_pred = mauna_test(x_train, y_train, x_test, k_max)
    return k_costs, y_pred, x_test # list of arrays holding costs for each k for all partitions


def maunua_cross_val(x_data : 'list[np.ndarray]', y_data : 'list[np.ndarray]', dist : Callable, k : int):
    '''
    output the total costs for each value of k in array
    construct queues for each point in the current validation set
    '''
    # take rmse of the costs for all points in validation set, for each value of k
    k_costs = []
    for a in range(len(x_data)):
        #pq_list : list[pq] = [] # one pq_list for each of the validation points
        costs = [0] * k    # separate validation set, training set
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
            x_val_i = x_val[i]
            y_val_i = y_val[i]

            for j in range(x_train.shape[0]):
                # to a single pq, add all points in the training set compared to a single validation point
                distance = dist(x_val_i, x_train[j])
                nq.put((distance, y_train[j]))
            short_nq = pq()
            for _ in range(k):
                short_nq.put(nq.get())

            #pq_list.append(short_nq) # only store the k nearest neighbors
            # get the errors for each value of k
            y_pred = 0
            for h in range(1,k+1):
                new_y = short_nq.get(block = False)
                # print(new_y[1])
                y_pred = y_pred *(h-1)/h + (new_y[1][0]) / h
                prev = costs[h-1]
                costs[h-1] = prev + (y_pred - y_val_i) ** 2
                #print(costs[h-1])
        for i in range(k):
            costs[i] = sqrt(costs[i] / x_val.shape[0]) # to find RMSE cost of each value of k for the validation set
        k_costs.append(costs)
    return k_costs

def mauna_test(x_train, y_train, x_test, k):
    # build the pqs (one for each point in test set)
    # for each, find y_value using a preset k (less efficient but will work nonetheless)
    # print(x_val.shape)
    y_predictions = []
    
    for i in range(x_test.shape[0]):
        nq = pq()        
        x_test_i = x_test[i]
        
        for j in range(x_train.shape[0]):
            # print(x_test_i, x_train[j])
            dist = l2_vals(x_test_i[0], x_train[j][0])
            nq.put((dist, y_train[j]))
            

        y_pred = 0
        cur_preds = []
        for h in range(1, k+1):
            new_y = nq.get(block = False)
            prev = y_pred
            y_pred = prev *((h-1)/h) + ((new_y[1]) / h)
            cur_preds.append(y_pred)
        y_predictions.append(cur_preds)
    y_predictions = np.array(y_predictions)
    return y_predictions



    


if __name__ == "__main__":
    start = time()
    costs, y_predicts, x_test = knn_mauna(40)
    # costs = np.array(costs)
    # print(costs)
    end = time()
    print("Runtime: " + str(end - start))
    sum_costs = costs[0]
    for i in range(1, len(costs)):
        sum_costs += costs[i]
    for i in range(len(sum_costs)):
        sum_costs[i] = sum_costs[i] / len(sum_costs)
    print(sum_costs)

    # plot the cross-validation loss across multiple k values
    """ plt.plot(np.array(range(1,41)), sum_costs/5)
    plt.title('Costs for Mauna Loa for various k')
    plt.xlabel('k value')
    plt.ylabel('average cost across 5-fold cross validation')
    plt.show()
    plt.clear() """
    # plot prediction on test set for various k values
    # plot the cross-validation loss across multiple k values


    """ for i in (1, 2, 5, 10, 20 ,39):
        y_guess = y_predicts[:,i]
        plt.plot(x_test, y_guess)
        plt.title('Predictions for testing set: k = ' + str(i))
        plt.xlabel('Input value')
        plt.ylabel('Output value')
        plt.show()
 """




