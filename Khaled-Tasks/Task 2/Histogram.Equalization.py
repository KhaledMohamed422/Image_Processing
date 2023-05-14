import cv2 as cv
import numpy as np

"""
# Step 1: Calculate the histogram
# Step 2: Calculate running sum over the histogram
# Step 3: Divide each value by the max value
# Step 4: Multiply by the new range
# Step 5: Replace the color of each pixel by the corresponding new color    
"""

def EqualizeHist(img):
    
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


img = cv.imread(
    r"D:\FCI\The 3 Level\The Second Term\New\Digital Image Processing\Tasks\imags\input_image1.png")


cv.imshow('output_image', EqualizeHist(img))
cv.waitKey(0)
cv.destroyAllWindows()
