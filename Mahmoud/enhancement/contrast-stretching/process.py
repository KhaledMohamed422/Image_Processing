import numpy as np
import cv2 as cv

def adjust_contrast(img, new_min, new_max):
    img_out = img.copy()
    rows, cols, channels = img_out.shape

    for ch in range(channels):
        old_min = np.min(img[:,:,ch])
        old_max = np.max(img[:,:,ch])
        
        if old_max - old_min == 0:
            continue
        
        for r in range(rows):
            for c in range(cols):
                old_value = img_out[r,c,ch]
                value = ((old_value - old_min)/(old_max - old_min))
                value = value * (new_max - new_min) + new_min
                
                img_out[r,c,ch] = value
    return img_out.astype(np.uint8)
  

org = cv.imread('org.jpg')
cv.imshow('original', org)
cv.waitKey(0)

contrast_enhanced = adjust_contrast(org, 0, 255)
cv.imshow('modified', contrast_enhanced)
cv.waitKey(0)
