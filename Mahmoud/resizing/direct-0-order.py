import cv2 as cv
import numpy as np

def direct_0_order(img, factor):
    new_rows = int(factor*img.shape[0])
    new_cols = int(factor*img.shape[1])

    resized_img = np.zeros((new_rows, new_cols, img.shape[2]), dtype=img.dtype)

    for channel in range(img.shape[2]):
        for r in range(img.shape[0]): 
            for c in range(img.shape[1]):
                resized_img[int(r*factor):int((r+1)*factor), int(c*factor):int((c+1)*factor), channel] = img[r, c, channel]
    
    return resized_img