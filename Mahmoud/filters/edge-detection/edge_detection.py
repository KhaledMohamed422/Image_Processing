#########################################
## Name:  Mahmoud Ahmed Zaytoni     #####
## Id:    2021425                   #####
## Group: G5                        #####
## assignment:  3                   #####
#########################################
import cv2 as cv
import numpy as np

def apply_edge_detection(img, mask):
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    rows,cols = gray_image.shape
    
    border_size = 3 // 2
    # Pad the input image with REPLICATE to handle borders
    padded_img = cv.copyMakeBorder(gray_image, border_size, border_size, border_size, border_size,cv.BORDER_REPLICATE)

    for i in range(border_size, rows + border_size):
        for j in range(border_size, cols + border_size):
            # Get the neighborhood of the current pixel
            neighborhood = padded_img[i-border_size:i+border_size+1, j-border_size:j+border_size+1]   
            sum = np.sum(neighborhood * mask, axis=(0,1))
            gray_image[i - border_size, j - border_size] = sum
    
    _, binary_image = cv.threshold(gray_image, 127, 255, cv.THRESH_BINARY)
    return binary_image

org = cv.imread('org.png')
# cv.imshow('original', org)
# cv.waitKey(0)

mask = np.array([[0, 1, 0],
                [1, -4, 1],
                [0, 1, 0]])

mask2 = np.array([[0, -1, 0],
                [-1, -4, -1],
                [0, -1, 0]])

mask3 = np.array([[1, 1, 1],
                [1, 8, 1],
                [1, 1, 1]])
    
mask4 = np.array([[-1, -1, -1],
                [-1, 8, -1],
                [-1, -1, -1]])

mask5 = np.array([[1, 2, 1],
                [0, 0, 0],
                [-1, -2, -1]])

out = apply_edge_detection(org, mask)
# cv.imshow('modified', out)
# cv.waitKey(0)

import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2)
axs[0].imshow(cv.cvtColor(org, cv.COLOR_BGR2RGB))
axs[0].set_title('Original Image')
axs[1].imshow(cv.cvtColor(out, cv.COLOR_BGR2RGB))
axs[1].set_title('Edge Detection')
plt.show()