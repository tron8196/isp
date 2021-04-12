# import cv2
from scipy.linalg import hadamard
import numpy as np
from scipy.io import loadmat

'''
This returns the type-2 DCT coefficient matrix for size N
'''
def getDCTCoeff(N, n, k):
    if k == 0:
        return np.round(1 / np.sqrt(N), 3)
    else:
        return np.round(np.sqrt(2 / N) * np.cos((np.pi * (2 * n + 1) * k) / (2 * N)), 3)

'''
Returns the amount of energy packed in first m coefficients
'''
def getEnergyPackingEfficiency(m, cov_matrix):
    return np.divide(np.diagonal(cov_matrix)[:m].sum(), np.trace(cov_matrix))*100


'''
Returns the de-correlation efficiency of a unitary transform
'''
def getDecorrelationEfficiency(R, R_dash):
    r_sum = np.sum(R) - np.trace(R)
    r_dash_sum = np.sum(np.abs(R_dash)) - np.trace(np.abs(R_dash))
    return (1 - np.divide(r_dash_sum, r_sum))*100

'''
This constructs a tri-diagonal matrix with param alpha
'''
def getQ(alpha, N):
  Q = [1 if i==j else -alpha if np.abs(i-j)==1 else 0 for i in range(N) for j in range(N)]
  Q = np.asarray(Q).reshape(N, N)
  Q[0,0] = 1-alpha
  Q[-1, -1] = 1-alpha
  return np.round(Q, 3)

'''
Get the hadamard matrix for 8*8 covariance matrix
'''
H = (1 / np.sqrt(8)) * hadamard(8)

rho = 0.95

'''
Setup Markov-1 covariance matrix
'''
covariance_matrix = np.asarray([rho ** np.abs(i - j) for i in range(8) for j in range(8)])
R = covariance_matrix.reshape(8, 8)

'''
De-correlate R by HRH' using the Hadamard Matrix  
'''
R_dash_Hadamard = np.matmul(np.matmul(H, R), H.T)

'''
Setup the DCT Matrix
'''
N = 8
D = np.array([getDCTCoeff(N, n, k) for k in range(N) for n in range(N)])
D = D.reshape(N, N)

'''
De-correlate R by DRD' using the DCT Matrix  
'''

R_dash_DCT = np.matmul(np.matmul(D, R), D.T)


eig_val_R, eig_vec_R = np.linalg.eig(R)

print('eig_vec_R', np.round(eig_vec_R, 3).T)
print()
print('DCT basis', D)
print()

'''
Find energy packing efficiency for DCT and Hadamard Matrix for first 4 coeff
'''
epe_dct = getEnergyPackingEfficiency(4, R_dash_DCT)
epe_hadamard = getEnergyPackingEfficiency(4, R_dash_Hadamard)
print('EPE DCT = ', epe_dct)
print('EPE Hadamard = ', epe_hadamard)


'''
Find De-correlation efficiency of DCT and Hadamard Matrix
'''
dce_dct = getDecorrelationEfficiency(R, R_dash_DCT)
dce_hadamard = getDecorrelationEfficiency(R, R_dash_Hadamard)
print('DCE DCT = ', dce_dct)
print('DCE Hadamard = ', dce_hadamard)


'''
Task 2
'''

#geta beta square
beta_square = np.divide(1-rho**2, 1+rho**2)
#R^(-1)*beta^2
R_1_beta_square = np.round(beta_square*np.linalg.inv(R), 2)
# print(R_1_beta_square)

#define alpha
alpha = np.divide(rho, 1+rho**2)


N = 8
#get Q
Q = getQ(alpha, N)

#De-correlate Q using DCT, DQD'
Q_DCT = np.matmul(np.matmul(D, Q), D.T)

#De-correlate R`= R^(-1)*beta^2 using DCT, DR`D'
R_1_beta_square_DCT = np.matmul(np.matmul(D, R_1_beta_square), D.T)


print('R`', R_1_beta_square)
print()
print('Q', Q)
print()
'''
De-correlated R` and Q
'''
print('Q_dash', np.round(Q_DCT, 2))
print()
print('R`_dash', np.round(R_1_beta_square_DCT, 2))

