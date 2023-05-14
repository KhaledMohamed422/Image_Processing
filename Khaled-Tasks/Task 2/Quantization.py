import cv2 as cv
import numpy as np

img = cv.imread(
    r"D:\FCI\The 3 Level\The Second Term\New\Digital Image Processing\Tasks\imags\input_image1.png")

k = 4
gray_levels = 2 ** k
gap = 256 / gray_levels
colors = np.arange(gap, 256 + gap, gap)
new_img = np.zeros_like(img)
img = np.clip(img, 0, 255)

for c in range(img.shape[2]):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):

            index = int(img[i, j, c] // gap)
            new_img[i, j, c] = colors[index]

cv.imwrite('output_image.png', new_img)
