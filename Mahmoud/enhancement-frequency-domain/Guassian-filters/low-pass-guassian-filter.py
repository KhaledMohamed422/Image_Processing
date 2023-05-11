import numpy as np
import cv2 as cv

img = cv.imread('org.png', 0)

def guassian_lowpass_filter(img, D0):
  fft = np.fft.fft2(img)
  fft_image_shifted = np.fft.fftshift(fft)

  real = np.real(fft_image_shifted)
  imaginary = np.imag(fft_image_shifted)

  m, n = img.shape[:2]
  center = (m // 2, n // 2)
  filter = np.zeros((m, n), dtype=np.float32)
  for i in range(m):
    for j in range(n):
      D = np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2)
      filter[i,j] = np.exp((-(D**2) / (2*D0**2)))

  real = real * filter
  imaginary = imaginary * filter

  # revers transform to get image
  filtered_fft_image = np.fft.ifftshift(real + 1j * imaginary)

  filterd_image = np.abs(np.fft.ifft2(filtered_fft_image))
  filterd_image = filterd_image / filterd_image.max() * 255 # normalize image
  return filterd_image.astype(np.uint8)

import matplotlib.pyplot as plt

# create a 2x2 subplot grid
fig, axs = plt.subplots(2, 2)

# plot the original image
axs[0, 0].imshow(img, cmap='gray')
axs[0, 0].set_title('Original')

# plot the filtered images
axs[0, 1].imshow(guassian_lowpass_filter(img, 5), cmap='gray')
axs[0, 1].set_title('D0 = 5')

axs[1, 0].imshow(guassian_lowpass_filter(img, 15), cmap='gray')
axs[1, 0].set_title('D0 = 15')

axs[1, 1].imshow(guassian_lowpass_filter(img, 30), cmap='gray')
axs[1, 1].set_title('D0 = 30')

fig.suptitle('Applying Gaussian Lowpass Filter')
# display the plot
plt.show()
