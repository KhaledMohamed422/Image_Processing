# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 12:57:49 2023

@author: SH-31
"""
import cv2 as cv
import numpy as np
from math import floor
img = cv.imread('img/2.jpeg')


def Edge_Detection(img,s):
    #converte img to gray level. 
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    
    #set size of the mask
    mask_size = s
    
    # formula for choice the padding size knowing the mask size 
    padded_s = floor(mask_size / 2)  
    
    # Pad the input image with REPLICATE to handle borders
    img_padded = cv.copyMakeBorder(img, padded_s, padded_s, padded_s, padded_s,cv.BORDER_REPLICATE)
    
    # Create mask for Sharpening Filter
    mask = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.int32) 
    
    #create new image to hold the edges
    img_Edge = np.zeros_like(img, dtype=np.uint8)

    # Apply the Edge Detection filter on the image
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_Edge[i, j] = np.sum(mask * img_padded[i:i+mask_size, j:j+mask_size])
            
            if img_Edge[i, j] < 0 :
                img_Edge[i, j] = 0
            elif img_Edge[i, j] > 255:
                img_Edge[i, j] = 255
    
    # Post-processing convert A gray image to a binery. 
    _, binary_img = cv.threshold(img_Edge,  127,255, cv.THRESH_BINARY)
    return binary_img

cv.imshow('Edge of the img',Edge_Detection(img))
cv.imshow('orgain',img)
cv.waitKey(0)
cv.destroyAllWindows()