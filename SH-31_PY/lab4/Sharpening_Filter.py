# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 12:57:49 2023

@author: SH-31
"""
import cv2 as cv
import numpy as np
from math import floor

img = cv.imread('img/1.jpg')
# img = cv.resize(img,(400,250))



def Sharpening_Filter(img):
    #set size of the mask
    mask_size = 3
    
    # formula for choice the padding size knowing the mask size 
    padded_s = floor(mask_size / 2) 
    
    # Pad the input image with REPLICATE to handle borders
    img_padded = cv.copyMakeBorder(img, padded_s, padded_s, padded_s, padded_s,cv.BORDER_REPLICATE)
    
    # Create mask for Sharpening Filter
    mask = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    # [[ -1, -1, -1],[ -1, 9, -1],[ -1, -1, -1]]
 
    #create new image
    Sharped_img = np.zeros_like(img, dtype=np.uint8)

    #Applying Sharpening Filter mask on the image
    for ch in range(img.shape[2]):  
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                value = 0
                
                for k in range(mask_size):
                    for l in range(mask_size):
                        value += img_padded[i+k, j+l, ch] * mask[k, l]
                
                # Post Processing Cut off 
                if value < 0 :
                    Sharped_img[i, j,ch] = 0
                elif value > 255:
                    Sharped_img[i, j,ch] = 255
                else:
                    Sharped_img[i, j,ch] = value
                    
    return Sharped_img            

cv.imshow('Sharpening Filter img',Sharpening_Filter(img))
cv.imshow('Orgain',img)
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
