import cv2 as cv
import numpy as np

img = cv.imread(
    r"D:\FCI\The 3 Level\The Second Term\New\Digital Image Processing\Tasks\imags\input_image1.png")

NewMin = 0
NewMax = 255

OldMin = np.min(img)
OldMax = np.max(img)

size = (NewMax - NewMin) / (OldMax - OldMin)

for c in range(img.shape[2]):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j, c] = ((img[i, j, c] - OldMin) * size) + NewMin

cv.imshow('output_image', img)
cv.waitKey(0)
cv.destroyAllWindows()
