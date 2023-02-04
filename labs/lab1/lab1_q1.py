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
'''
x train, x valid, x test, y train, y valid, y test = load dataset('mauna loa')
x train, x valid, x test, y train, y valid, y test = load dataset('rosenbrock', n train=5000, d=2)
x train, x valid, x test, y train, y valid, y test = load dataset('pumadyn32nm')
x train, x valid, x test, y train, y valid, y test = load dataset('iris')
x train, x valid, x test, y train, y valid, y test = load dataset('mnist small')
'''

# l1 is manhattan dist, l2 is euclidian dist

def l1(vec1 : np.ndarray, vec2 : np.ndarray):
    assert vec1.shape() == vec2.shape(), "cannot compute distance for vectors of different dimensions"
    distance = 0
    for i in range(vec1.shape[0]):
        distance += abs(vec1[i] - vec2[i])
    return distance

def l2(vec1 : np.ndarray, vec2 : np.ndarray):
    assert vec1.shape() == vec2.shape(), "cannot compute distance for vectors of different dimensions"
    distance = 0
    for i in range(vec1.shape[0]):
        distance += (vec1[i] - vec2[i])**2
    return sqrt(distance)


def randomize(array_x : np.ndarray, array_y: np.ndarray, split_axis : int = 1):
    assert array_x.shape == array_y.shape
    joined = np.concatenate((array_x, array_y), axis=split_axis)
    
    np.random.shuffle(joined)
    [shuf_x, shuf_y] = np.split(joined, 2, axis = 1)
    for i in range(3):
        print(shuf_x[i], shuf_y[i])



# to keep variable naming and code readable, perform regression for each dataset in individual functions
def knn_mauna():
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('mauna_loa') # output is np arrays
    # use single set for validations and training, and a separate test for testing (once hyperparameters have been chosen)
    x_train, y_train = np.vstack([x_train, x_valid]), np.vstack([y_train, y_valid])
    
    assert x_train.shape == y_train.shape, "training data shape invalid"
    

    # want even multiples of five, discarding an insignificant amount of data
    while(x_train.shape[0] %5):
        x_train = x_train[:-1]
    while(y_train.shape[0] %5):
        y_train = y_train[:-1]
    for i  in range (3):
        print(x_train[i], y_train[i])

    [xt1, xt2, xt3, xt4, xt5] = np.split(x_train, 5, axis = 0)
    randomize(x_train, y_train, split_axis = 1)

    # five separate sets of data, 








if __name__ == "__main__":
    knn_mauna()