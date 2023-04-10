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

def power_low_transformation(img, gamma):
    transformed_img = img.astype(np.longlong)

    transformed_img = transformed_img**gamma
    
    return adjust_contrast(transformed_img, 0, 255) # Normalization

org = cv.imread('org.jpg')
cv.imshow('original', org)
cv.waitKey(0)

out = power_low_transformation(org, 5)
cv.imshow('modified', out)
cv.waitKey(0)