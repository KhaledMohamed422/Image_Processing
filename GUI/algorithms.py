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

def histogram_equalization(img):
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
    # cv.imshow('org', img)
    # cv.waitkey(0)
    print('hellow')
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
        
    img_out = img_out.astype(np.uint8)

    img_out = Image.fromarray(cv.cvtColor(img_out, cv.COLOR_BGR2RGB))
    return img_out

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
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
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
                   
              
    new_img = Contrast(new_img, 0, 255) 
    new_img = Image.fromarray(cv.cvtColor(new_img, cv.COLOR_BGR2RGB))
    return new_img
