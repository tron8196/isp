import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

class NLM:

    def __init__(self):
        self.latent_image = None
        self.noisy_image = None
        self.n_rows = self.n_cols = None

    def readLatentAndNoisyImage(self, latent_img_path, noisy_img_path):
        self.latent_image = (cv2.imread(latent_img_path).astype(np.float32))/255
        self.noisy_image = (cv2.imread(noisy_img_path).astype(np.float32))/255
        self.n_rows, self.n_cols, _ = self.latent_image.shape

    def calculatePSNR(self, f_hat):
        f_vec = self.latent_image.flatten()
        f_hat_vec = f_hat.flatten()
        diff = f_vec - f_hat_vec
        MSE = np.dot(diff, diff)/self.latent_image.size
        PSNR = 10 * np.log10(255*255/MSE)
        return PSNR

    def gaussianFilter(self, sigma, ksize=7):
        kernel = cv2.getGaussianKernel(ksize=ksize, sigma=sigma)
        return cv2.filter2D(src=self.noisy_image, ddepth=cv2.CV_32F, kernel=kernel)

    def nlmFilter(self, W, Wsim, sigma_nlm):
        g = np.pad(self.noisy_image, ((Wsim + W, Wsim + W), (Wsim + W, Wsim + W), (0, 0)), 'constant')
        f_hat = np.zeros(self.noisy_image.shape)
        for row_index in range(Wsim + W, self.n_rows + Wsim + W):
            for col_index in range(Wsim + W, self.n_cols + Wsim + W):

                Np = g[row_index - Wsim:row_index + Wsim + 1, col_index - Wsim:col_index + Wsim + 1, :]

                Vp = Np.flatten()

                wp = np.zeros((2 * W + 1, 2 * W + 1))

                for patch_row_index in range(-W, W + 1):
                    for patch_col_index in range(-W, W + 1):
                        Nq = g[row_index + patch_row_index - Wsim:row_index + patch_row_index + Wsim + 1,
                               col_index + patch_col_index - Wsim:col_index + patch_col_index + Wsim + 1, :]

                        Vq = Nq.flatten()

                        diff_vec = Vp - Vq
                        wp[patch_row_index + W, patch_col_index + W] = np.exp(-np.dot(diff_vec, diff_vec) / (sigma_nlm ** 2))

                wp = wp / sum(sum(wp))

                wp = wp.flatten()

                NpWR = g[row_index - W:row_index + W + 1, col_index - W:col_index + W + 1, 0]
                VpWR = NpWR.flatten()

                NpWG = g[row_index - W:row_index + W + 1, col_index - W:col_index + W + 1, 1]
                VpWG = NpWG.flatten()

                NpWB = g[row_index - W:row_index + W + 1, col_index - W:col_index + W + 1, 2]
                VpWB = NpWB.flatten()

                f_hat[row_index - Wsim - W, col_index - Wsim - W, 0] = np.dot(VpWR, wp)
                f_hat[row_index - Wsim - W, col_index - Wsim - W, 1] = np.dot(VpWG, wp)
                f_hat[row_index - Wsim - W, col_index - Wsim - W, 2] = np.dot(VpWB, wp)
        return f_hat


nlm = NLM()
nlm.readLatentAndNoisyImage('./krishna.png', './krishna_0_001.png')

W = 3
Wsim = 3
PSNR_list = []
for sigma in np.arange(0.1, 0.6, 0.1):
    print('Finding PSNR for sigma '+str(sigma))
    f_hat = nlm.nlmFilter(Wsim=Wsim, W=W, sigma_nlm=sigma)
    PSNR_list.append(nlm.calculatePSNR(f_hat))
print(PSNR_list)

plt.plot(np.arange(0.1, 1.1, 0.1), PSNR_list, color='r', label='W=3')


W = 5
Wsim = 3
PSNR_list = []
for sigma in np.arange(0.1, 1.1, 0.1):
    print('Finding PSNR for sigma '+str(sigma))
    f_hat = nlm.nlmFilter(Wsim=Wsim, W=W, sigma_nlm=sigma)
    PSNR_list.append(nlm.calculatePSNR(f_hat))
print(PSNR_list)


plt.plot(np.arange(0.1, 0.6, 0.1), PSNR_list, color='g', label='W=5')
plt.legend()
plt.grid()
plt.show()

