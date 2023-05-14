import cv2 as cv
import numpy as np


img = cv.imread(
    r"D:\FCI\The 3 Level\The Second Term\New\Digital Image Processing\imags\image24.png")


mask_size = 41
mask = np.ones((mask_size, mask_size), dtype=np.float32) / \
    (mask_size * mask_size)

output_img = np.zeros_like(img)
size = mask_size // 2
for i in range(size, img.shape[0] - size):
    for j in range(size, img.shape[1] - size):
        for k in range(3):
            average = np.sum(img[i - size:i + size + 1,
                             j - size:j + size + 1, k] * mask)
            output_img[i, j, k] = average

cv.imshow("Orginal", img)
cv.imshow("Output_img", output_img)
cv.waitKey(0)
cv.destroyAllWindows()
