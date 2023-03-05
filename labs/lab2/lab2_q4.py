from data.data_utils import load_dataset # include copy of data folder in submission zip
import numpy as np
from time import time
import math
from matplotlib import pyplot as plt
from typing import List

# ========== data analysis/plotting ============
class plotter_1d:
    def __init__(self, ds : str = 'mauna_loa', ms = 1):
        x_train, x_valid, self.x_test, y_train, y_valid, self.y_test = load_dataset(ds)

        # to ensure independence from testing set, plot only validation and training sets
        self.x_plot, self.y_plot = np.vstack([x_train, x_valid]), np.vstack([y_train, y_valid])
        self.ms = ms
    
    def show(self):
        plt.xlabel('input value')
        plt.ylabel('output value')
        # plt.legend(['Model Prediction', "Test Data"])
        plt.show()
    
    def plot_train(self):
        plt.plot(self.x_plot, self.y_plot, marker = 'o', markersize =self.ms, color = (0,0,1), linestyle = None)
    def plot_more(self, x, y):
        plt.plot(x, y,  color = (1,0,0))
    def plot_test(self):
        plt.plot(self.x_test, self.y_test, marker = 'o', markersize = self.ms, color = (0,0,1), linestyle = None, linewidth = 0)

# ======= for performing MDL =========:
# func classes: 'sin', 'lin', 'quad', 'hor'
'''
sin: params = [A, omega, phi]
lin: params = [m]
exp: params = [B, c]
hor: params = [d]
'''

# ================= the basis functions class ==============
class basis_func:
    def __init__(self, params, func_class, input_vec: np.ndarray):
        self.params = params
        self.func_class = func_class
        N = np.shape(input_vec)[0]
        self.phi = np.zeros((N, 1))
        self.phi = self.evalute(input_vec)
    
    # use vectorized version
    def evalute(self, x):
        if self.func_class == 'sin':
            return self.params[0] * np.sin((self.params[1] * x - self.params[2]))
        elif self.func_class == 'lin':
            return self.params[0] * x
        elif self.func_class == 'hor':
            return np.full(np.shape(x), self.params[0])
        elif self.func_class == 'quad':
            return self.params[0] * np.square(x - np.full(np.shape(x), self.params[1]))
        else:
            raise Exception('invalid function class')
    
    def __repr__(self):
        return self.func_class + ': ' + str(self.params)
    
    def __eq__(self, other):
        return (self.func_class == other.func_class) and (self.params == other.params)


# ============ Class for Orthogonal matching pursuit ===========
class OMP:
    def select_cand(self) -> basis_func:
        '''
        select a candidate from one of the remaining candidates based on J values
        '''
        if len(self.cands) == 0:
            raise Exception('no more candidates')
            return
        j_max, argmax = 0, self.cands[0]
        elem : basis_func
        elem = self.cands[0]
        j_max = find_j(elem.phi, self.r)

        for elem in self.cands:
            cur_j = find_j(elem.phi, self.r)
            if cur_j > j_max:
                j_max = cur_j
                argmax = elem
        self.cands.remove(argmax)
        self.selec.append(argmax)
        return argmax
    
    def run_omp(self):
        '''
        method is called once setup is complete, runs iterations of OMP algorithm
        until the stopping criterion's value is minimized (i.e. it begins to increase again)
        '''
        self.omp_iter()
        while self.stop_crit() < self.eps:
        # for _ in range(2):
            self.eps = self.stop_crit()
            self.omp_iter()
        return self.w, self.Phi
    
    def omp_iter(self):
        '''
        performs a single iteration of OMP
        '''
        if self.k > self.md:
            raise Exception('reached depth before acceptable error')
        self.k += 1
        new_selec : basis_func = self.select_cand()
        print(self.k, new_selec)
        if not np.any(self.Phi):
            self.Phi = new_selec.phi
        else:
            self.Phi = np.hstack([self.Phi, new_selec.phi]) # append it?
        # print(np.shape(self.Phi))
        
        self.w = np.linalg.pinv(self.Phi).dot(self.y_train)
        self.r = self.Phi.dot(self.w) - self.y_train
        print(self.stop_crit(), self.eps)

    def stop_crit(self):
        '''
        helper function to evaluate stopping criterion value
        '''
        # print(np.shape(self.Phi), np.shape(self.w), np.shape(self.y_train))
        loss = np.linalg.norm((self.Phi).dot(self.w) - self.y_train, 2)
        return (self.N/2) * math.log(loss) + (self.k/2) * math.log(self.N)
    
    def test(self):
        '''
        function to compare extrapolation of model to test data and print the RMSE of the test
        '''
        # for each data point in the test set, evaluate estimated output as sum of output of all selected candidates
        y_pred = np.zeros(np.shape(self.x_test))
        for i in range(len(self.selec)):
            cand = self.selec[i]
            re_eval = basis_func(cand.params, cand.func_class, self.x_test)
            y_pred += self.w[i] * (re_eval.phi)
        self.test_pred = y_pred
        print(math.sqrt(np.square(self.test_pred -self.y_test).mean()))

    def __init__(self, args, dataset, eps, max_depth, merge_sets = False,):
        assert(dataset == 'mauna_loa')
        self.x_train, self.x_valid, self.x_test, self.y_train, self.y_valid, self.y_test = load_dataset(dataset)
        
        if merge_sets:
            self.x_train, self.y_train = np.vstack([self.x_train, self.x_valid]), np.vstack([self.y_train, self.y_valid])
        
        self.cands, self.N = build_basis(args, input_vec = self.x_train)
        self.selec = []
        self.k = 0
        self.r = self.y_train # initializing r_0
        self.N = self.x_train.size
        self.eps = 1000000 # if first prediction equals zero
        self.md = max_depth
        self.Phi = np.zeros(np.shape(self.x_train))
        self.w = np.zeros((np.shape(self.x_train)[0],1)) # arbitrary


def build_basis(args, input_vec):
    '''
    builds a dictionary of basis functions using the arguments passed in
    arguments are passed in the following fashion:
    args[i] : [func_class, (max param_i, min param_i, step_size_i)]
    '''
    basis  : List[basis_func] = []
    size = 0
    for arg in args:
        func_class = arg[0]
        if func_class == 'sin':
            A, A_max, A_step = arg[1][0], arg[1][1], arg[1][2]
            omega, omega_max, omega_step = arg[2][0], arg[2][1], arg[2][2]
            phi, phi_max, phi_step = arg[3][0], arg[3][1], arg[3][2]
            while A <= A_max:
                omega = arg[2][0]
                while omega <= omega_max:
                    phi = arg[3][0]
                    while phi <= phi_max:
                        basis.append(basis_func([A, omega, phi], 'sin', input_vec))
                        size += 1
                        phi += phi_step
                    omega += omega_step
                A += A_step
        if func_class == 'lin':
            m, m_max, m_step = arg[1][0], arg[1][1], arg[1][2]
            while m <= m_max:
                basis.append(basis_func([m], 'lin', input_vec))
                size+= 1
                m += m_step

        if func_class == 'quad':
            a, a_max, a_step = arg[1][0], arg[1][1], arg[1][2]
            b, b_max, b_step = arg[1][0], arg[1][1], arg[1][2]
            while a < a_max:
                b = arg[1][0]
                while b < b_max:
                    basis.append(basis_func([a, b], 'quad', input_vec))
                    size += 1
                    b += b_step
                a += a_step
        if func_class == 'hor':
            d, d_max, d_step = arg[1][0], arg[1][1], arg[1][2]
            while d <= d_max:
                basis.append(basis_func([d], 'hor', input_vec))
                size+= 1
                d += d_step

    return basis, size

"""  if func_class == 'exp':
    b, b_max, b_step = arg[1][0], arg[1][1], arg[1][2]
    c, c_max, c_step = arg[2][0], arg[2][1], arg[2][2]
    while b <= b_max:
        while c <= c_max:
            basis.append(basis_func([b,c], 'exp', input_vec))
            size+= 1
            c += c_step
        b += b_step """



def find_j(phi_i : np.ndarray, resid : np.ndarray):
    '''
    helper function to evaluate J for a given phi and residual
    '''
    return ((np.transpose(phi_i).dot(resid))**2) / (np.matmul(np.transpose(phi_i), phi_i))


if __name__ == '__main__':
    '''
    main block instantiates the OMP object and performs all necessary tests
    '''
    args = [
        ['sin', [0.05, 0.4, 0.05],  [100, 115, 1], [math.pi/2, 3*math.pi/2, math.pi/ 16]],
        ['lin', [0.6, 1.4, 0.01]],
        ['quad', [0.1, 0.5, 0.05], [-2, -1, 0.05]],
        ['hor', [-0.2, 0, 0.01]]
    ]
    start = time()
    optim = OMP(args, 'mauna_loa', 200, 10000, merge_sets=True)
    
    # print(optim.cands)
    print(len(optim.cands), 'basis functions in dictionary, after ' + str(time() - start) + 's')

    weights, kernels = optim.run_omp()
    optim.test()
    
    plotter = plotter_1d(ms = 5)
    pred = kernels.dot(weights)
    # plotter.plot_more(optim.x_train, pred)
    plotter.plot_more(optim.x_test, optim.test_pred)
    plotter.plot_test()
    plotter.show()

