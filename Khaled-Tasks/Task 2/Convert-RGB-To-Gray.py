import cv2 as cv
import numpy as np

img = cv.imread(
    r"D:\FCI\The 3 Level\The Second Term\New\Digital Image Processing\Tasks\imags\input_image1.png")


gray = img[:, :, 1]

cv.imshow('output_image', gray)
cv.waitKey(0)
cv.destroyAllWindows()
