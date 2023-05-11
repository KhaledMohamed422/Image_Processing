#########################################
## Name:  Mahmoud Ahmed Zaytoni     #####
## Id:    2021425                   #####
## Group: G5                        #####
## assignment:  3                   #####
#########################################
import numpy as np
import cv2 as cv
import math

def create_gaussian_filter_mask(sigma):
    N = math.floor(3.7 * sigma - 0.5)
    mask_size = 2 * N + 1
    t = mask_size // 2

    # Create a 2D Gaussian filter mask
    gaussian_mask = np.zeros((mask_size, mask_size), dtype=np.float64)
    for i in range(mask_size):
        for j in range(mask_size):
            x = i - t
            y = j - t
            gaussian_mask[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)

    # Normalize the mask
    gaussian_mask /= np.sum(gaussian_mask)

    return gaussian_mask

def apply_gaussian_filter(img, sigma):
    rows, cols, channels = img.shape
    output = np.zeros((rows, cols, channels), dtype=np.uint8)
    
    gaussian_mask = create_gaussian_filter_mask(sigma)
    mask_size, _ = gaussian_mask.shape
    
    border_size = mask_size // 2
    
    padded_img = cv.copyMakeBorder(img, border_size, border_size, border_size, border_size,cv.BORDER_REPLICATE).astype('float64')
    
    for i in range(border_size, rows + border_size):
        for j in range(border_size, cols + border_size):
            pixel_sum = np.zeros(channels, dtype=np.float64)
            
            for k in range(-border_size, border_size + 1):
                for l in range(-border_size, border_size + 1):
                    pixel_sum += padded_img[i + k, j + l]*gaussian_mask[k+border_size, l+border_size]
            
            output[i - border_size, j - border_size] = pixel_sum
    print('finish')
    return output

def apply_gaussian_filter_builtin(img, sigma):
    # Create a 1D Gaussian kernel for the rows
    size, _ = create_gaussian_filter_mask(sigma).shape
    kernel_1d = cv.getGaussianKernel(size, sigma)
    
    # Create a 2D Gaussian kernel by taking the outer product of the row kernel with itself
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    
    # Normalize the kernel so that its elements sum to 1
    kernel_2d /= np.sum(kernel_2d)
    # Apply the filter using OpenCV's filter2D function
    output = cv.filter2D(img, -1, kernel_2d)
    return output


org = cv.imread('org.png')
# cv.imshow('original', org)
# cv.waitKey(0)

# out = apply_gaussian_filter(org, 0.7)
# cv.imshow('modified', out)
# cv.waitKey(0)

out = apply_gaussian_filter_builtin(org, 3)
# cv.imshow('modified', out)
# cv.waitKey(0)

import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2)
axs[0].imshow(cv.cvtColor(org, cv.COLOR_BGR2RGB))
axs[0].set_title('Original Image')
axs[1].imshow(cv.cvtColor(out, cv.COLOR_BGR2RGB))
axs[1].set_title('Gaussian with sigma = 3 Image')
plt.show()