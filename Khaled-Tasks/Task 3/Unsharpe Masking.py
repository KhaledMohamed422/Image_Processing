import cv2 as cv
import numpy as np

img = cv.imread(
    r"D:\FCI\The 3 Level\The Second Term\New\Digital Image Processing\imags\man.png")

gaussian_blur = cv.GaussianBlur(img, (5, 5), 0)
unsharp_mask = cv.subtract(img, gaussian_blur)
output_img = cv.add(img, unsharp_mask)

cv.imshow("Original", img)
cv.imshow("Output_img", output_img)
cv.waitKey(0)
cv.destroyAllWindows()
