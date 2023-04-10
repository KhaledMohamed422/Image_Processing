import cv2 as cv
import numpy as np

img = cv.imread('image/gamma.jpg')


def Brightness(img,offset):
   
    if len(img.shape) == 2 :
        size = img.shape
        new_img = np.zeros(size, np.uint8)    
        for r in range(size[0]):
            for c in range(size[1]):
                new_val = img[r,c] + offset
                if new_val > 255: new_val = 255
                if new_val < 0  : new_val = 0
                new_img[r,c] = new_val     
    
    else:
        size = img.shape
        new_img = np.zeros(size, np.uint8)    
        for r in range(size[0]):
            for c in range(size[1]):
                for ch in range(size[2]):
                    new_val = img[r,c,ch] + offset
                    if new_val > 255: new_val = 255
                    if new_val < 0  : new_val = 0
                    new_img[r,c,ch] = new_val     
              
    return new_img
    
def Contrast(img,new_min,new_max):
   
    if len(img.shape) == 2 :
        size = img.shape
        new_img = np.zeros(size, np.uint8) 
    
        old_max = np.amax(img)
        old_min = np.amin(img)
        for r in range(size[0]):
            for c in range(size[1]):
                new_val = ((img[r,c] - old_min)/(old_max-old_min)*(new_max-new_min)+new_min)
                if new_val > 255: new_val = 255
                if new_val < 0  : new_val = 0
                new_img[r,c] = new_val     
    
    else:
        size = img.shape
        new_img = np.zeros(size, np.uint8)
        old_max = np.amax(img)
        old_min = np.amin(img)
        for r in range(size[0]):
            for c in range(size[1]):
                for ch in range(size[2]):
                    new_val = ( ((img[r,c,ch] - old_min) / (old_max-old_min) ) * (new_max-new_min) + new_min)
                   
                    if new_val > 255: new_val = 255
                    if new_val < 0  : new_val = 0
                    new_img[r,c,ch] = new_val     
              
    return new_img


def Power_Law_Transformations(img,gamma):
   
    if len(img.shape) == 2 :
        size = img.shape
        new_img = np.zeros(size)    
        for r in range(size[0]):
            for c in range(size[1]):
                new_val = img[r,c] ** gamma
                new_img[r,c] = new_val     
                    
    else:
        size = img.shape
        new_img = np.zeros(size)    
        for r in range(size[0]):
            for c in range(size[1]):
                for ch in range(size[2]):
                    new_val = img[r,c,ch] ** gamma                  
                    new_img[r,c,ch] = new_val   
                   
              
    return Contrast(new_img, 0, 255)

cv.imshow('Power_Law_Transformations',Power_Law_Transformations(img,0.5))
cv.imshow('orgain',img)
cv.waitKey(0)    