import scipy.fft as f
import numpy as np
import cmath
import cv2
import matplotlib.pyplot as plt


def createImageFromPhaseAndMagnitude(fft_mag, fft_phase):
    fft_img = np.multiply(fft_mag, np.exp(complex(0, 1) * fft_phase))
    intermediate_image = f.ifft(fft_img, axis=1)
    img = np.abs(f.ifft(intermediate_image, axis=0))
    return img


class FFT2D:
    source_img = n_rows = n_cols = size = None
    source_img_fft = None
    source_img_fft_mag = source_img_fft_phase = None

    def readImage(self, image_path):
        self.source_img = cv2.imread(image_path, 0)
        self.n_rows, self.n_cols = self.source_img.shape
        self.size = self.n_rows * self.n_cols
        print('Image Read Successful')

    def plotFftMag(self):
        y = np.linspace(0, self.n_rows - 1, self.n_rows)
        x = np.linspace(0, self.n_cols - 1, self.n_cols)
        X, Y = np.meshgrid(x, y)
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, self.source_img_fft_mag, cmap='viridis', edgecolor='none')
        plt.show()

    def find2DFFT(self):
        intermediate_fft = f.fft(self.source_img, axis=1)
        self.source_img_fft = f.fft(intermediate_fft, axis=0)
        self.source_img_fft_mag = np.abs(self.source_img_fft)
        self.source_img_fft_phase = np.angle(self.source_img_fft)

    def find2DCenteredFFt(self):
        y = np.linspace(0, self.n_rows - 1, self.n_rows)
        x = np.linspace(0, self.n_cols - 1, self.n_cols)
        X, Y = np.meshgrid(x, y)
        freq_shifted_source_img = np.multiply(self.source_img, np.power(-1, X + Y))
        intermediate_fft = f.fft(freq_shifted_source_img, axis=0)
        self.source_img_fft = f.fft(intermediate_fft, axis=1)
        self.source_img_fft_mag = np.abs(self.source_img_fft)
        self.source_img_fft_phase = np.angle(self.source_img_fft)

    def getMagAndPhaseFFT(self):
        return self.source_img_fft_mag, self.source_img_fft_phase

    def writeFFTMagPlot(self, f_name='test_mag_plot'):
        t_img = np.round(np.log10(1 + self.source_img_fft_mag)).astype(np.uint8)
        t_img = (t_img / np.max(t_img)) * 255
        cv2.imwrite('./output/'+f_name+'.png', t_img)


a = FFT2D()
a.readImage('fourier_transform.png')
a.find2DCenteredFFt()
img1_mag, img1_phase = a.getMagAndPhaseFFT()
a.writeFFTMagPlot('img1_mag')

a.readImage('fourier.png')
a.find2DCenteredFFt()
img2_mag, img2_phase = a.getMagAndPhaseFFT()
a.writeFFTMagPlot('img2_mag')


img_mag1_phase2 = createImageFromPhaseAndMagnitude(img1_mag, img2_phase)
img_mag2_phase1 = createImageFromPhaseAndMagnitude(img2_mag, img1_phase)

cv2.imwrite('./output/img_mag1_phase2.png', img_mag1_phase2)
cv2.imwrite('./output/img_mag2_phase1.png', img_mag2_phase1)















