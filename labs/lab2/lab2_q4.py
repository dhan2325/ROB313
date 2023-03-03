from data.data_utils import load_dataset # include copy of data folder in submission zip
import numpy as np
from time import time
import math
from matplotlib import pyplot as plt
from typing import List

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

# ======= for performing MDL =========:
# func classes: 'sin', 'lin', 'exp', 'hor'
'''
sin: params = [A, omega, phi]
lin: params = [m]
exp: params = [B, c]
hor: params = [d]
'''
class basis_func:
    def __init__(self, params, func_class, input_vec):
        self.params = params
        self.func_class = func_class
        N = np.shape(input_vec)[0]
        self.phi = np.zeros((N, 1))
        for i in range(N):
            self.phi[i] = self.evalute(input_vec[i])
    
    def evalute(self, x):
        if self.func_class == 'sin':
            return self.params[0] * math.sin(math.radians(self.params[1] * x - self.params[2]))
        elif self.func_class == 'lin':
            return self.params[0] * x
        elif self.func_class == 'exp':
            return self.params[0] * math.exp(self.params[1] * x)
        elif self.func_class == 'hor':
            return self.params[0]
        else:
            raise Exception('invalid function class')
    
    def __repr__(self):
        return self.func_class + ': ' + str(self.params)
    
    def __eq__(self, other):
        return (self.func_class == other.func_class) and (self.params == other.params)


class OMP:
    def __init__(self, args, dataset, eps, max_depth, merge_sets = False,):
        assert(dataset == 'mauna_loa')
        self.x_train, self.x_valid, self.x_test, self.y_train, self.y_valid, self.y_test = load_dataset(dataset)
        
        if merge_sets:
            self.x_train, self.y_train = np.vstack([self.x_train, self.x_valid]), np.vstack([self.y_train, self.y_valid])
        
        self.cands, self.N = build_basis(args, input_vec = self.x_train)
        self.selec = []
        self.k = 0
        self.r = self.y_train # initializing r_0
        self.eps = eps
        self.md = max_depth
        self.Phi = None
        self.w = np.zeros((np.shape(self.x_train)[0],1))
    
    def select_cand(self):
        if len(self.cands) == 0:
            raise Exception('no more candidates')
            return
        j_min, argmin = None, None
        elem : basis_func
        for elem in self.cands:
            if (j_min == None):
                j_min = find_j(elem.phi, self.r)
                argmin = elem
            else:
                cur_j = find_j(elem.phi, self.r)
                if cur_j < j_min:
                    j_min = cur_j
                    argmin = elem
        self.cands.remove(argmin)
        self.selec.append(argmin)
        return argmin
    
    def run_omp(self):
        while np.linalg.norm(self.w, 2) > self.eps:
            if self.k > self.md:
                assert('reached depth before acceptable error')
            self.k += 1
            new_selec = self.select_cand()
            if self.Phi == None:
                self.Phi = new_selec
            else:
                self.Phi = np.vstack(self.Phi, new_selec)
            self.w = np.linalg.solve(self.phi, self.y_train) # i think we look for approximation and not solution? this will not work

        return self.w, self.selec


# args[i] : [func_class, (max param_i, min param_i, step_size_i)]
def build_basis(args, input_vec):
    basis  : List[basis_func] = []
    size = 0
    for arg in args:
        func_class = arg[0]
        if func_class == 'sin':
            A, A_max, A_step = arg[1][0], arg[1][1], arg[1][2]
            omega, omega_max, omega_step = arg[2][0], arg[2][1], arg[2][2]
            phi, phi_max, phi_step = arg[3][0], arg[3][1], arg[3][2]
            while A <= A_max:
                print(A)
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
    """ numer = (np.transpose(phi_i) * resid)**2
    denom = np.matmul(np.transpose(phi_i), phi_i) """
    return ((np.transpose(phi_i) * resid)**2) / (np.matmul(np.transpose(phi_i), phi_i))

def stop_crit(N,  ls_loss, k):
    return (N/2) * math.log(ls_loss) + (k/2) * math.log(N)

if __name__ == '__main__':
    """ plotter = plotter_1d()
    plotter.show()
 """
    # TODO: set values for these parameters!
    # no exponential for now. If linear sucks, we'll try exponential
    # 1259 basis functions in about 2.5 seconds, can adjust granularity later
    args = [
        ['sin', [0.05, 0.2, 0.025],  [2*math.pi/0.08, 2*math.pi/0.04, 5], [math.pi/2, 3*math.pi/2, math.pi/ 10]],
        ['lin', [0.6, 1.4, 0.05]],
        ['hor', [-0.2, 0, 0.02]],
    ]
    start = time()
    optim = OMP(args, 'mauna_loa', 0, 0, merge_sets=False)
    
    # print(optim.cands)
    print(len(optim.cands), 'basis functions in dictionary, after ' + str(time() - start) + 's')

