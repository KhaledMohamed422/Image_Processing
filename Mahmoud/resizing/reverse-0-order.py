import cv2 as cv
import numpy as np

def reverse_0_order(img, output_size): # (height, width)
    # Compute scaling ratios
    row_ratio = img.shape[0] / output_size[0] # height
    col_ratio = img.shape[1] / output_size[1] # width

    # Compute output image using direct mapping with 0-order
    resized_img = np.zeros((output_size[0], output_size[1], img.shape[2]), dtype=img.dtype)

    for channel in range(img.shape[2]):
        for new_x in range(output_size[0]):
            old_x = round(new_x * row_ratio)
            for new_y in range(output_size[1]):
                old_y = round(new_y * col_ratio)
                resized_img[new_x, new_y, channel] = img[old_x, old_y, channel]
    
    return resized_img

def reverse_0_order_builtin(img, output_size):
    # Resize image using nearest-neighbor interpolation
    return cv.resize(img, output_size, interpolation=cv.INTER_NEAREST)
