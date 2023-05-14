import cv2
import numpy as np

img = cv2.imread(
    r"D:\FCI\The 3 Level\The Second Term\New\Digital Image Processing\Tasks\imags\input_image1.png")


hist = np.zeros((256, 1), dtype=np.uint8)

for row in range(img.shape[0]):
    for col in range(img.shape[1]):

        pixel_value = img[row, col]
        hist[pixel_value] += 1

print(hist)
