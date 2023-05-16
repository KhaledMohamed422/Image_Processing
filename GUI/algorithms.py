import cv2 as cv
import numpy as np
from PIL import Image

def adjust_brightness_optimized(img, offset):
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
    # Cast the input image to int16 data type to allow for negative pixel values
    img_int = img.astype(np.int16)
    
    # Add the offset to the image pixel values
    img_int += offset
    
    # Clip the pixel values to the range of 0-255
    img_clipped = np.clip(img_int, 0, 255)
    
    # Cast the output image to uint8 data type to ensure pixel values are in the range of 0-255
    img_out = img_clipped.astype(np.uint8)
    img_out = Image.fromarray(cv.cvtColor(img_out, cv.COLOR_BGR2RGB))
    return img_out

def cvt2gray_luminance(img):
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
    gray_img = np.round(0.3*img[:,:,0] + 0.59*img[:,:,1] + 0.11*img[:,:,2]).astype(np.uint8)
    img_out = Image.fromarray(cv.cvtColor(gray_img, cv.COLOR_BGR2RGB))
    return img_out
