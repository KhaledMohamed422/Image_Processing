import cv2 as cv
import numpy as np

# Load the two images
img1 = cv.imread(
    r"D:\FCI\The 3 Level\The Second Term\New\Digital Image Processing\Tasks\imags\input_image1.png")
img2 = cv.imread(
    r"D:\FCI\The 3 Level\The Second Term\New\Digital Image Processing\Tasks\imags\input_image1.png")

img2 = cv.resize(img2, img1.shape[:2][::-1])  # resize img2 to match img1

NewImage = np.zeros(img1.shape, dtype=np.uint8)

for c in range(img1.shape[2]):
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):

            NewImage[i, j, c] = np.clip(
                int(img1[i, j, c]) - int(img2[i, j, c]), 0, 255)

cv.imwrite("output_image.jpg", NewImage)
