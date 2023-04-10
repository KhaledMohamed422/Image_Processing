import cv2 as cv
import numpy as np

def interpolate_pixels(arr, factor):
    indices = np.arange(0, len(arr), factor)
    vals = np.take(arr, indices)

    for i in range(1, len(indices)):
        start_idx = indices[i-1]
        end_idx = indices[i]

        start_val = vals[i-1]
        end_val = vals[i]

        diff = end_val - start_val
        for j in range(start_idx+1, end_idx):
            # actually this equation for each two cases: --> max > min and min > max
            arr[j] = round((diff/factor) * (j-start_idx)) + start_val

    last_index = indices[-1]
    arr[last_index:] = arr[last_index]
         
def direct_1_order(img, factor):
    new_rows = int(factor*img.shape[0])
    new_cols = int(factor*img.shape[1])
    
    resized_img = np.zeros((new_rows, new_cols, img.shape[2]))
    
    for channel in range(img.shape[2]):
        for r in range(img.shape[0]):
            for c in range(img.shape[1]):
                resized_img[int(r*factor), int(c*factor), channel] = img[r, c, channel]
        
        for r in range(resized_img.shape[0]):
            interpolate_pixels(resized_img[r, :, channel], factor)
        for c in range(resized_img.shape[1]):
            interpolate_pixels(resized_img[:, c, channel], factor)

    return resized_img.astype('uint8')