# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 12:57:49 2023

@author: SH-31
"""
import cv2 as cv
import numpy as np
from math import floor


img = cv.imread('img/1.jpg')


def Edge_Detection(img):
    #converte img to gray level. 
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    
    #set size of the mask
    mask_size = 3
    
    # formula for choice the padding size knowing the mask size 
    padded_s = floor(mask_size / 2)  
    
    # Pad the input image with REPLICATE to handle borders
    img_padded = cv.copyMakeBorder(img, padded_s, padded_s, padded_s, padded_s,cv.BORDER_REPLICATE)
    
    # Create mask for Sharpening Filter
    mask = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype=np.int32) 
    
    #create new image to hold the edges
    img_Edge = np.zeros_like(img, dtype=np.int64)

    # Apply the Edge Detection filter on the imag
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_Edge[i, j] = np.sum(mask * img_padded[i:i+mask_size, j:j+mask_size])
            
            if img_Edge[i, j] < 0 :
                img_Edge[i, j] = 0
            elif img_Edge[i, j] > 255:
                img_Edge[i, j] = 255
    
    # Post-processing convert A gray image to a binery. 
    _, binary_img = cv.threshold(img_Edge.astype('uint8'),  100,255, cv.THRESH_BINARY)
    return binary_img

cv.imshow('Edge of the img',Edge_Detection(img))
# cv.imshow('edge build in',edges)
cv.imshow('orgain',img)
cv.waitKey(0)
cv.destroyAllWindows()          



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
    # Create mask for Weighted_Filter
    mask,mask_size = gaussian_mask(sigma)
    
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



def Unsharpening_Filter(img):
        
    # Create a blur image as intger 16 to do image algebra. 
    blur_img = Smoothing_Weighted_Filter(img,2).astype('int16')
    
    #Subtract the image with blur img to get edge of the image.
    edeg_img = np.clip(img - blur_img, 0, 255).astype('int16')
   
    #Adding the image original image with edge.  
    image = np.clip(edeg_img + img, 0, 255).astype('uint8')
    return image
    

cv.imshow('Unsharpening_Filter img',Unsharpening_Filter(img))
cv.imshow('orgain',img)
cv.waitKey(0)
cv.destroyAllWindows()
