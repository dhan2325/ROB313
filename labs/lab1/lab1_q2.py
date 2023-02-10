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
    # print(array_x.shape, array_y.shape)
    joined = np.concatenate((array_x, array_y), axis=split_axis)
    
    np.random.shuffle(joined)
    [shuf_x, shuf_y] = np.split(joined, [2], axis = 1)
    #print(shuf_x.shape, shuf_y.shape)
    return shuf_x, shuf_y


# make class to keep all data accessible in main block
class rosenbrock:
    def __init__(self): # import dataset
        self.x_train, self.x_valid, self.x_test, self.y_train, self.y_valid, self.y_test = load_dataset(\
            'rosenbrock', n_train=1000, d=2)
        self.y_lookup = {}
    

    def setup_knn(self, max_k : int = 10):
        self.x_train, self.y_train = np.vstack([self.x_train, self.x_valid]), np.vstack([self.y_train, self.y_valid])
        self.k_max = max_k
        # assert x_train.shape[0] == y_train.shape[0], "training data shape invalid"
        self.x_train, self.y_train = randomize(self.x_train, self.y_train)

        # want even multiples of five, discarding an insignificant amount of data
        while(self.x_train.shape[0] %5):
            self.x_train = self.x_train[:-1]
        while(self.y_train.shape[0] %5):
            self.y_train = self.y_train[:-1]
        
        self.x_partit = np.split(self.x_train, 5, axis = 0)
        self.y_partit = np.split(self.y_train, 5, axis = 0)

        
        self.k_costs = np.array(self.cross_val_kdt(l2_vec))

    def cross_val_kdt(self, dist : Callable):
        '''
        output the total costs for each value of k in array
        construct queues for each point in the current validation set
        '''
        # take rmse of the costs for all points in validation set, for each value of k
        self.kd_trees = []
        self.k_costs = []
        
        self.y_lookup = {}
        # create a single dictionary of all points in all datasets
        for i in range(len(self.x_partit)):
            for j in range(len(self.x_partit[i])):
                self.y_lookup[tuple(self.x_partit[i][j])] = self.y_partit[i][j]
        
        # print(y_lookup)
        for a in range(len(self.x_partit)):
            # form a kdtree for every partition
            kd_tree = kdt(self.x_partit[a])
            self.kd_trees.append(kd_tree)
            # print(kd_tree)
            costs = [0] * self.k_max

            # sort partitions for the current iteration
            x_val, y_val = self.x_partit[a], self.y_partit[a]
            # print(x_val.shape)
            if (a == 0):
                b = 1
            else:
                b = 0
            x_train, y_train = self.x_partit[b], self.y_partit[b]
            for i in range(len(self.x_partit)):
                if (i!=a) and (i != b):
                    x_train, y_train = np.vstack([x_train, self.x_partit[i]]), np.vstack([y_train, self.y_partit[i]])
            
        

if __name__ == "__main__":
    rosen = rosenbrock()
    rosen.setup_knn(max_k = 10)
    [dist, indices] = rosen.kd_trees[0].query([(0,0)], k = 5)
    print(indices[0])
    for index in indices[0]:
        print(rosen.x_partit[0][index], rosen.y_lookup[tuple(rosen.x_partit[0][index])], '\n ')
        #print(type(rosen.x_partit[0][index]))
    
    