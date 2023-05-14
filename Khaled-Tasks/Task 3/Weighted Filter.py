import cv2 as cv
import numpy as np


img = cv.imread(
    r"D:\FCI\The 3 Level\The Second Term\New\Digital Image Processing\imags\image24.png")

# Compute mask size using standard deviation
sigma = 10
N = int(3.7 * sigma - 0.5)
mask_size = 2 * N + 1

# Fill Gaussian mask
x, y = np.meshgrid(np.arange(-N, N+1), np.arange(-N, N+1))
gaussian_mask = (1 / (2 * np.pi * sigma**2)) * \
    np.exp(-(x**2 + y**2) / (2 * sigma**2))

# Normalize mask so its sum equals 1
gaussian_mask = gaussian_mask / np.sum(gaussian_mask)

output_img = np.zeros_like(img)
size = mask_size // 2
for i in range(size, img.shape[0] - size):
    for j in range(size, img.shape[1] - size):
        for k in range(3):
            average = np.sum(
                img[i - size:i + size + 1, j - size:j + size + 1, k] * gaussian_mask)
            output_img[i, j, k] = average

cv.imshow("Orginal", img)
cv.imshow("Output_img", output_img)
cv.waitKey(0)
cv.destroyAllWindows()
