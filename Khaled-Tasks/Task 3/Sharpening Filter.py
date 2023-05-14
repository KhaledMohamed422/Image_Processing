import cv2 as cv
import numpy as np


img = cv.imread(
    r"D:\FCI\The 3 Level\The Second Term\New\Digital Image Processing\imags\input_imageIndex2.png")

# Define sharpening mask
sharpen_mask = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

output_img = np.zeros_like(img)
size = 1  

for i in range(size, img.shape[0] - size):
    for j in range(size, img.shape[1] - size):
        for k in range(3):
            center = img[i, j, k]
            neighbors = img[i - size:i + size + 1, j - size:j + size + 1, k]
            laplacian = np.sum(sharpen_mask * neighbors) 
            output_img[i, j, k] = np.clip(5 * center - laplacian, 0, 255)

cv.imshow("Orginal", img)
cv.imshow("Output_img", output_img)
cv.waitKey(0)
cv.destroyAllWindows()
