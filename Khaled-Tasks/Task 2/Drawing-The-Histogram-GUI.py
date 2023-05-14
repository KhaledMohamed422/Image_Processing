import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
img = cv2.imread(
    r"D:\FCI\The 3 Level\The Second Term\New\Digital Image Processing\Tasks\imags\input_image1.png")

# Initialize the histogram matrix
hist = np.zeros((256, 1), dtype=np.uint8)

# Loop through each pixel in the image and update the histogram matrix
for row in range(img.shape[0]):
    for col in range(img.shape[1]):
        pixel_value = img[row, col]
        hist[pixel_value] += 1

# Display the histogram as a bar chart
plt.bar(np.arange(256), hist.ravel())
plt.show()
