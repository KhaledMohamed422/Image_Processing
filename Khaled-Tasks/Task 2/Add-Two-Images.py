import cv2 as cv
import numpy as np

img1 = cv.imread(
    r"D:\FCI\The 3 Level\The Second Term\New\Digital Image Processing\Tasks\imags\input_image1.png")
img2 = cv.imread(
    r"D:\FCI\The 3 Level\The Second Term\New\Digital Image Processing\Tasks\imags\input_image1.png")

img2 = cv.resize(img2, img1.shape[:2][::-1])  # resize img2 to match img1

NewImage = np.zeros_like(img1)

for c in range(img1.shape[2]):
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):

            NewImage[i, j, c] = img1[i, j, c] + img2[i, j, c]

cv.imwrite('output_image.jpg', NewImage)
