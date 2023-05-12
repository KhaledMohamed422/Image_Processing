import numpy as np
import cv2 as cv

def histogram_equalization(img):
    img_out = img.copy()
    rows, cols, channels = img.shape

    for ch in range(channels):
        # Create a list of 256 zeros
        colors = [0] * 256
        for r in range(rows):
            for c in range(cols):
                colors[img[r,c,ch]] += 1

        for i in range(1, len(colors)): # Range Sum (or prefix sum)
            colors[i] += colors[i-1]
            
        for i in range(len(colors)):
            colors[i] /= colors[-1] # division by last element of range sum array
            colors[i] = round(255*colors[i])

        for r in range(rows):
            for c in range(cols):
                img_out[r,c,ch] = colors[img[r,c,ch]]
        
    return img_out.astype(np.uint8)

org = cv.imread('org.jpg')
cv.imshow('original', org)
cv.waitKey(0)

out = histogram_equalization(org)
cv.imshow('modified', out)
cv.waitKey(0)