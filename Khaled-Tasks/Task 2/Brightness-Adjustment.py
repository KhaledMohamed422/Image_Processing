import cv2 as cv
import numpy as np

img = cv.imread(
    r"D:\FCI\The 3 Level\The Second Term\New\Digital Image Processing\Tasks\imags\input_image1.png")

offset = int(input("Enter size of offset : "))
NewImage = np.zeros_like(img)

for c in range(img.shape[2]):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            new_val = np.clip(img[i, j, c] + offset, 0, 255)
            NewImage[i, j, c] = new_val

cv.imshow('output_image', NewImage)
cv.waitKey(0)
cv.destroyAllWindows()
