from scipy.io import loadmat
import numpy as np



'''
Find the frobenius norm of (g-g_hat)
'''
def getFrobeneiusNorm(g, g_hat):
    return np.linalg.norm(g-g_hat)**2


g = loadmat('imageFile.mat').get('g')

A, sigma, B_H = np.linalg.svd(g, compute_uv=True)
#
rank_g = sigma[np.abs(sigma) > 0].size

'''
Reconstruct g by using all the non-zero singular values of g
'''
g_hat = np.zeros(g.shape)
for i in range(rank_g):
    g_hat = g_hat + sigma[i] * np.matmul(A[:, [i]], B_H[[i], :])

print('g_hat', np.round(g_hat, 2))
print('g', g)
print('norm', getFrobeneiusNorm(g, g_hat))


print()
'''
g by using first m non-zero singular values of g, here m=4
'''
m=7
g_hat = np.zeros(g.shape)
print()
print('Using only '+ str(m) +' singular values')

for i in range(m):
    g_hat = g_hat + sigma[i] * np.matmul(A[:, [i]], B_H[[i], :])


print()
print('g_hat', np.round(g_hat, 2))
print('g', g)
print('norm', getFrobeneiusNorm(g, g_hat))
print('singular_square', np.square(sigma)[:m].sum())


m=6
g_hat = np.zeros(g.shape)
print()
print('Using only '+ str(m) +' singular values')

for i in range(m):
    g_hat = g_hat + sigma[i] * np.matmul(A[:, [i]], B_H[[i], :])


print()
print('g_hat', np.round(g_hat, 2))
print('g', g)
print('norm', getFrobeneiusNorm(g, g_hat))
print('singular_square', np.square(sigma)[:m].sum())

m=3
g_hat = np.zeros(g.shape)
print()
print('Using only '+ str(m) +' singular values')

for i in range(m):
    g_hat = g_hat + sigma[i] * np.matmul(A[:, [i]], B_H[[i], :])


print()
print('g_hat', np.round(g_hat, 2))
print('g', g)
print('norm', getFrobeneiusNorm(g, g_hat))
print('singular_square', np.square(sigma)[:m].sum())
