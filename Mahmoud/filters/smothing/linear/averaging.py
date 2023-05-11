#########################################
## Name:  Mahmoud Ahmed Zaytoni     #####
## Id:    2021425                   #####
## Group: G5                        #####
## assignment:  3                   #####
#########################################
import cv2 as cv
import numpy as np

def apply_mean_filter(img, mask_size):
    rows, cols, channels = img.shape
    output = np.zeros((rows, cols, channels), dtype=np.uint8)
    
    border_size = mask_size // 2
    
    # Pad the input image with REPLICATE to handle borders
    padded_img = cv.copyMakeBorder(img, border_size, border_size, border_size, border_size,cv.BORDER_REPLICATE)
    
    mask = np.ones((mask_size, mask_size), dtype=np.float32)

    # Normalize the mask
    mask /= np.sum(mask)
    for i in range(border_size, rows + border_size):
        for j in range(border_size, cols + border_size):
            pixel_sum = np.zeros(channels, dtype=np.uint32)
            for k in range(-border_size, border_size + 1):
                for l in range(-border_size, border_size + 1):
                    pixel_sum += padded_img[i + k, j + l]
            mean = pixel_sum // (mask_size * mask_size)
            output[i - border_size, j - border_size] = mean
    
    return output

def apply_mean_filter_builtin(img, mask_size):
    # Create a mask filled with ones
    mask = np.ones((mask_size, mask_size), dtype=np.float32)

    # Normalize the mask
    mask /= np.sum(mask)

    # Apply the filter using OpenCV's filter2D function
    output = cv.filter2D(img, -1, mask)
    return output

org = cv.imread('org.png')
# cv.imshow('original', org)
# cv.waitKey(0)

# out = apply_mean_filter(org, 5)
# cv.imshow('modified', out)
# cv.waitKey(0)

out = apply_mean_filter_builtin(org, 9)
# cv.imshow('modified', out)
# cv.waitKey(0)

import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2)
axs[0].imshow(cv.cvtColor(org, cv.COLOR_BGR2RGB))
axs[0].set_title('Original Image')
axs[1].imshow(cv.cvtColor(out, cv.COLOR_BGR2RGB))
axs[1].set_title('Averaging Image')
plt.show()