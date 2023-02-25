from data.data_utils import load_dataset # include copy of data folder in submission zip
import numpy as np
from math import sqrt
from typing import Callable, List
from queue import PriorityQueue as pq
from time import time
from sklearn.neighbors import KDTree as kdt
import matplotlib.pyplot as plt

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
# for question 2, no cross-validation needed: add validation data to training set, use k = 5
# no need to randomize/split dataset
class rosenbrock:
    def __init__(self, ls : int = 40, d_val = 2): # import dataset
        self.x_train, self.x_valid, self.x_test, self.y_train, self.y_valid, self.y_test = load_dataset(\
            'rosenbrock', n_train=5000, d=d_val)
        self.y_lookup = {}
        self.leaf_size = 40
    

    def setup_knn(self):
        self.x_train, self.y_train = np.vstack([self.x_train, self.x_valid]), np.vstack([self.y_train, self.y_valid])
        self.build_kdt()

    def build_kdt(self):
        '''
        output the total costs for each value of k in array
        construct queues for each point in the current validation set
        '''
        self.k_costs = []
        self.y_lookup = {}
        self.kd_tree = kdt(self.x_train, leaf_size = self.leaf_size) # keep default at 40


    def get_nn(self, k_set, points : List[tuple]):
        # return two arrays: one for all the x-coords of the nearest neighbours, 
        # another for the y-coords of those nearest neighbours
        # TODO: WRITE FUNCTION TO GET K NEAREST NEIGHBOURS
        [dist, indices]  = self.kd_tree.query(points, k = k_set)
        x_points = []
        y_points = []
        for ind in indices:
            x_points.append(tuple(self.x_train[ind]))
            y_points.append(self.y_train[ind]) # since the points are all in order, no need for dict?
        return x_points, y_points
            
        

if __name__ == "__main__":
    d_vals = 30
    times = []

    rosen = rosenbrock(d_val = 2)
    rosen.setup_knn()
    x, y = rosen.get_nn(5, np.random.random((5, 2)))

    for i in range(3, d_vals):
        start = time()
        rosen = rosenbrock(d_val=i)
        rosen.setup_knn()
        print(type(rosen.x_test))
        x, y = rosen.get_nn(5, [[0] * i])
        print(x[0][0], '\n', y[0][0])
        times.append(time()-start)
    
    plt.plot(range(3, d_vals), times)
    plt.xlabel('Dimensionality of Input')
    plt.ylabel('Overall Runtime')
    plt.title('Runtimes for kNN with kd-trees')
    plt.show()
    
    
    