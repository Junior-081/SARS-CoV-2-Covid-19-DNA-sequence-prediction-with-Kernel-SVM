import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import pandas as pd
import cvxopt
import warnings
warnings.filterwarnings("ignore" )

from utils import error,plot_decision_function
from model import rbf_kernel, sigma_from_median,linear_kernel,polynomial_kernel,KernelMethodBase

# Utilities
def cvxopt_qp(P, q, G, h, A, b):
    P = .5 * (P + P.T)
    cvx_matrices = [
        cvxopt.matrix(M) if M is not None else None for M in [P, q, G, h, A, b] 
    ]
    #cvxopt.solvers.options['show_progress'] = False
    solution = cvxopt.solvers.qp(*cvx_matrices, options={'show_progress': False})
    return np.array(solution['x']).flatten()

solve_qp = cvxopt_qp


#Import of DataSet
X = np.loadtxt('data/Xtr.csv', skiprows=1, usecols=(1,), dtype=str, delimiter=',')
y = np.loadtxt('data/Ytr.csv', skiprows=1, usecols=(1,), dtype=int, delimiter=',')
Xtest = np.loadtxt('data/Xte.csv', skiprows=1, usecols=(1,), dtype=str, delimiter=',')

X_tr_vector = np.loadtxt('data/Xtr_vectors.csv', skiprows=1, usecols=(), dtype=str, delimiter=',')
X_test_vector = np.loadtxt('data/Xte_vectors.csv', skiprows=1, usecols=(), dtype=str, delimiter=',')

X_full = np.hstack([X, Xtest])
y = 2*y - 1.

#Break up a sequence into subsequences

def get_kmers(sequence, kmer_size=4):
    return [sequence[i: i+kmer_size] for i in range(len(sequence) - kmer_size + 1)]

def base2int(c):
    return {'A': 0, 'C': 1, 'G': 2, 'T': 3}.get(c, 0)

def index(kmer):
    # Transform the kmer into sequence of character indices
    return [base2int(c) for c in kmer]

def spectral_embedding(sequence):
    kmers = get_kmers(sequence)
    multiplier = 4 ** np.arange(len(kmer))[::-1]
    kmer_indices = [multiplier.dot(index(kmer)) for kmer in kmers]
    one_hot_vector = np.zeros(4**kmer_size).astype(int)
    for kmer_index in kmer_indices:
        one_hot_vector[kmer_index] += 1
    return one_hot_vector

kmer_size = 4
kmer = X[0][0:kmer_size] 
base_indices = np.array([base2int(base) for base in kmer])
# base_indices
multiplier = 4 ** np.arange(len(kmer))
# multiplier
kmer_index = multiplier[::-1].dot(base_indices)
# print(kmer_index)

# Enconding of X_train
liste2=[]
for i in X:
    sequence = i
    liste2.append(spectral_embedding(sequence))
l=np.array(liste2)
X_train=l/(len(sequence)-kmer_size+1)

# print(X_train)

# print("----------------------------------------------------------------")

# Enconding of X_test

kmer2 = Xtest[0][0:kmer_size] 
base_indices2 = np.array([base2int(base) for base in kmer2])
# base_indices
multiplier2 = 4 ** np.arange(len(kmer2))
# multiplier
kmer_index2 = multiplier2[::-1].dot(base_indices2)

def spectral_embedding_test(sequence2):
    kmers = get_kmers(sequence2)
    multiplier = 4 ** np.arange(len(kmer2))[::-1]
    kmer_indices = [multiplier.dot(index(kmer2)) for kmer2 in kmers]
    one_hot_vector = np.zeros(4**kmer_size).astype(int)
    for kmer_index in kmer_indices:
        one_hot_vector[kmer_index] += 1
    return one_hot_vector

liste_=[]
for j in Xtest:
    sequence2 = j
    liste_.append(spectral_embedding_test(sequence2))
d=np.array(liste_)
X_test=d/(len(sequence2)-kmer_size+1)

# print(X_test)
#Split y_train and y_test, you can shuffle y 

y_train,y_test=y,y[1000:]

#All values are converted to float

X_train=X_train.astype(float) 
y_train=y_train.astype(float) 
X_test=X_test.astype(float) 


#KERNEL SVM

#Soft-margin dual problem

def svm_dual_soft_to_qp_kernel(K, y, C=1):
    n = K.shape[0]
    assert (len(y) == n)
        
    # Dual formulation, soft margin
#     P = np.diag(y).dot(K).dot(np.diag(y))
    P=K*y*y[:,None]
    # As a regularization, we add epsilon * identity to P
    eps = 1e-12
    P += eps * np.eye(n)
    q = - np.ones(n)
    G = np.vstack([-np.eye(n), np.eye(n)])
    h = np.hstack([np.zeros(n), C * np.ones(n)])
    A = y[np.newaxis, :]
    b = np.array([0.])
    return P, q, G, h, A, b

K = linear_kernel(X_train, X_train)
alphas = solve_qp(*svm_dual_soft_to_qp_kernel(K, y_train, C=1.))

#-----------------------

class KernelSVM(KernelMethodBase):
    '''
    Kernel SVM Classification
    
    Methods
    ----
    fit
    predict
    '''
    def __init__(self, C=0.1, **kwargs):
        self.C = C
        super().__init__(**kwargs)
    
    def fit_K(self, K, y, tol=1e-3):
        # Solve dual problem
        self.alpha = solve_qp(*svm_dual_soft_to_qp_kernel(K, y, C=self.C))
        
        # Compute support vectors and bias b
        sv = np.logical_and((self.alpha > tol), (self.C - self.alpha > tol))
        self.bias = y[sv] - K[sv].dot(self.alpha * y)
        self.bias = self.bias.mean()

        self.support_vector_indices = np.nonzero(sv)[0]
        
        return self
        
    def decision_function_K(self, K_x):
        return K_x.dot(self.alpha * self.y_train) + self.bias

    def predict(self, X):
        return np.sign(self.decision_function(X))

#Train our Kernel SVM
kernel = 'rbf'
sigma = 1.
degree = 3
C =200
tol = 1e-1
model = KernelSVM(C=C, kernel=kernel, sigma=sigma, degree=degree)

#Test our Kernel SVM

y_pred = model.fit(X_train, y_train, tol=tol).predict(X_test)
# plot_decision_function(model, X_test, y_test,title='SVM {} Kernel'.format(kernel))
print(" ")
print('Test error: {:.2%}'.format(error(y_pred, y_test)))

print(" ")

pred = np.where(y_pred == -1, 0, 1) #To convert all -1 values to 0

print("PREDICTION ")
print("------------------------")

print(pred)



Id=(np.arange(1000)+1).reshape(-1,1) #Creation of indexes
y_save = np.c_[Id,pred] #Concatenate Indexes and y_pred
# y_save[:10]

# Save as a csv file
np.savetxt('result/sample_prediction.csv', y_save,
           delimiter=',', header='Id,Covid', fmt='%i', comments='')