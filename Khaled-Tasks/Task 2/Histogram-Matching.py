import cv2 as cv
import numpy as np

"""
1-Histogram equalize the two images.
2-Match the equalized histogram of the 1st image with the one of the 2nd image.
3- Replace the color of each pixel by the corresponding new match color 
"""

def EqualizeHist(img):

    if len(img.shape) == 2:
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    NewImage = np.zeros_like(img)

    for c in range(img.shape[2]):

        # Step 1: Calculate the histogram
        h, bins = np.histogram(img[:, :, c], bins=256, range=(0, 255))

        # Step 2: Calculate running sum over the histogram
        cdf = np.cumsum(h)

        # Step 3: Divide each value by the max value
        cdf_normalized = cdf / cdf.max()

        # Step 4: Multiply by the new range
        new_range = 255
        equalized = np.round(cdf_normalized * new_range)

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                # Step 5: Replace the color of each pixel by the corresponding new color
                new_val = equalized[img[i, j, c]]
                NewImage[i, j, c] = new_val

    return NewImage


img1 = cv.imread(
    r"D:\FCI\The 3 Level\The Second Term\New\Digital Image Processing\Tasks\imags\input_image1.png")
img2 = cv.imread(
    r"D:\FCI\The 3 Level\The Second Term\New\Digital Image Processing\Tasks\imags\input_image1.png")

gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

equalizeHistGray1 = EqualizeHist(gray1)
equalizeHistGray2 = EqualizeHist(gray2)

h1, b1 = np.histogram(equalizeHistGray1.flatten(), 256, [0, 256])
h2, b2 = np.histogram(equalizeHistGray2.flatten(), 256, [0, 256])

cdf1 = h1.cumsum()
cdf2 = h2.cumsum()

cdf1 = (cdf1 / cdf1.max()) * 255
cdf2 = (cdf2 / cdf2.max()) * 255

lut = np.interp(cdf1, cdf2, range(256))

NewImage = np.zeros_like(img1)

for c in range(img1.shape[2]):
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):

            new_val = int(lut[equalizeHistGray1[i, j, c]])
            NewImage[i, j, c] = new_val

cv.imwrite('output_image.jpg', NewImage)
