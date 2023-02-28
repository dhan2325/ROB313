from data.data_utils import load_dataset # include copy of data folder in submission zip
import numpy as np
from time import time
import math
from matplotlib import pyplot as plt

# ========== data analysis ============
class plotter_1d:
    def __init__(self, ds : str = 'mauna_loa'):
        x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset(ds)

        # to ensure independence from testing set, plot only validation and training sets
        self.x_plot, self.y_plot = np.vstack([x_train, x_valid]), np.vstack([y_train, y_valid])
    
    def show(self):
        plt.plot(self.x_plot, self.y_plot, 'o', markersize = 2)
        plt.xlabel('input data')
        plt.ylabel('output data')
        plt.show()


if __name__ == '__main__':
    plotter = plotter_1d()
    plotter.show()