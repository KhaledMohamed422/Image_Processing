"""
Created on Mon Apr 10 19:45:07 2023

@author: SH-31
"""
import cv2 as cv
import numpy as np
from math import floor

img = cv.imread('img/1.jpeg')
img = cv.resize(img,(600,450))
def Smoothing_Mean_Filter(img,mask_size:int):
    
    # formula for choice the padding size knowing the mask size 
    padded_s = round((mask_size - 1) / 2) + 1
    # Pad the input image with REPLICATE to handle borders
    img_padded = cv.copyMakeBorder(img, padded_s, padded_s, padded_s, padded_s,cv.BORDER_REPLICATE)
    
    # Create mask for mean filter
    mask = np.ones((mask_size,mask_size), dtype=np.int32) 
    # Apply mean filter by convolving kernel with image
    img_smoothed = np.zeros_like(img, dtype=np.uint8)

    for ch in range (img.shape[2]):  
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
              
                img_smoothed[i, j,ch] = np.sum(mask * img_padded[i:i+mask_size, j:j+mask_size,ch]) // (mask_size*mask_size)

    
    return img_smoothed


cv.imshow('Smoothing by Mean Filter',Smoothing_Mean_Filter(img,3))
cv.imshow('orgain',img)
cv.waitKey(0)
cv.destroyAllWindows()
'''

 other way to build gaussian_mask using build in functions
kernel_1 = cv.getGaussianKernel(3, 0.5)
kernel_2 = np.transpose(kernel_1)
# Multiply the two kernels to get a 3x3 Gaussian kernel
gaussian_kernel = kernel_1 * kernel_2

'''

def gaussian_mask(sigma):
    # Determine kernel size
    N = int(np.floor(3.7 * sigma - 0.5))
    size = 2 * N + 1
    # Create kernel
    t = np.arange(-N, N + 1)
    x, y = np.meshgrid(t, t)
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel /= np.sum(kernel)
    
    return kernel,size

def Smoothing_Weighted_Filter(img,sigma): 
    # Create mask for Weighted_Filter
    mask , mask_size = gaussian_mask(sigma)
    
    # Determine pannding size
    padded_s = floor(mask_size / 2) 
   
    # Pad the input image with REPLICATE to handle borders
    img_padded = cv.copyMakeBorder(img, padded_s, padded_s, padded_s, padded_s,cv.BORDER_REPLICATE)
    
    # Apply mean filter by convolving kernel with image
    img_smoothed = np.zeros_like(img, dtype=np.uint8)

    for ch in range (img.shape[2]):  
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                img_smoothed[i, j,ch] = np.sum(mask * img_padded[i:i+mask_size, j:j+mask_size,ch]) 

    return img_smoothed

sigma = 2
cv.imshow('smoothed_img by Weighted Filter',Smoothing_Weighted_Filter(img,2))
cv.imshow('orgain',img)
cv.waitKey(0)
cv.destroyAllWindows()