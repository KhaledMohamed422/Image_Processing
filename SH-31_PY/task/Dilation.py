import numpy as np
import cv2 as cv


img = cv.imread('img/F1.bmp',0)
# Apply thresholding to convert to binary
_, img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)

se = np.array([
    [1,1,1,1,1],
    [1,1,1,1,1],
    [1,1,1,1,1],
    [1,1,1,1,1],
    [1,1,1,1,1]
    ])
# se = np.ones((5,5))

def Dilation(img , se):
   
    Dilation_img = np.zeros_like(img)
    
    r , c = img.shape
    k_r , k_c = se.shape
    
    for i in range(k_r//2, r-k_r//2):
        for j in range(k_c//2, c-k_c//2):
            
            image_patch = img[i-k_r//2 :(i+k_r//2)+1 , j-k_c//2 :(j+k_c//2)+1 ]
            
            
            apply_and = np.logical_and(image_patch , se)
            Dilation_img[i,j] = np.max(apply_and)
            
         
    return Dilation_img
      

 

cv.imshow('orgin',img )
cv.imshow('Dilation img',Dilation(img,se) * 255) # open-cv can not show binary image so I muliply 1s by 255 to make the pixsl white in gray
cv.waitKey(0)
