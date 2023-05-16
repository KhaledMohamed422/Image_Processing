import cv2 as cv
import numpy as np
from math import floor
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


def gaussian_mask(sigma):
    # Determine kernel size
    N = int(np.floor(3.7 * sigma - 0.5))
    size = 2 * N + 1
    # Create kernel
    t = np.arange(-N, N + 1)
    x, y = np.meshgrid(t, t)
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel /= np.sum(kernel)
    
    return kernel,size

def Smoothing_Weighted_Filter(img,sigma): 

    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
    # Create mask for Weighted_Filter
    mask , mask_size = gaussian_mask(sigma)
    
    # Determine pannding size
    padded_s = floor(mask_size / 2) 
   
    # Pad the input image with REPLICATE to handle borders
    img_padded = cv.copyMakeBorder(img, padded_s, padded_s, padded_s, padded_s,cv.BORDER_REPLICATE)
    
    # Apply mean filter by convolving kernel with image
    img_smoothed = np.zeros_like(img, dtype=np.uint8)

    for ch in range (img.shape[2]):  
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                img_smoothed[i, j,ch] = np.sum(mask * img_padded[i:i+mask_size, j:j+mask_size,ch]) 

    img_smoothed = Image.fromarray(cv.cvtColor(img_smoothed, cv.COLOR_BGR2RGB))
    return img_smoothed


def Edge_Detection(img):

    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
    #converte img to gray level. 
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    
    #set size of the mask
    mask_size = 3
    
    # formula for choice the padding size knowing the mask size 
    padded_s = floor(mask_size / 2)  
    
    # Pad the input image with REPLICATE to handle borders
    img_padded = cv.copyMakeBorder(img, padded_s, padded_s, padded_s, padded_s,cv.BORDER_REPLICATE)
    
    # Create mask for Sharpening Filter
    mask = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.int32) 
    
    #create new image to hold the edges
    img_Edge = np.zeros_like(img, dtype=np.uint8)

    # Apply the Edge Detection filter on the image
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_Edge[i, j] = np.sum(mask * img_padded[i:i+mask_size, j:j+mask_size])
            
            if img_Edge[i, j] < 0 :
                img_Edge[i, j] = 0
            elif img_Edge[i, j] > 255:
                img_Edge[i, j] = 255
    
    # Post-processing convert A gray image to a binery. 
    _, binary_img = cv.threshold(img_Edge,  127,255, cv.THRESH_BINARY)
    binary_img = Image.fromarray(cv.cvtColor(binary_img, cv.COLOR_BGR2RGB))
    return binary_img




def Sharpening_Filter(img):
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
    #set size of the mask
    mask_size = 3
    
    # formula for choice the padding size knowing the mask size 
    padded_s = floor(mask_size / 2) 
    
    # Pad the input image with REPLICATE to handle borders
    img_padded = cv.copyMakeBorder(img, padded_s, padded_s, padded_s, padded_s,cv.BORDER_REPLICATE)
    
    # Create mask for Sharpening Filter
    mask = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    # [[ -1, -1, -1],[ -1, 9, -1],[ -1, -1, -1]]
 
    #create new image
    Sharped_img = np.zeros_like(img, dtype=np.uint8)

    #Applying Sharpening Filter mask on the image
    for ch in range(img.shape[2]):  
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                value = 0
                
                for k in range(mask_size):
                    for l in range(mask_size):
                        value += img_padded[i+k, j+l, ch] * mask[k, l]
                
                # Post Processing Cut off 
                if value < 0 :
                    Sharped_img[i, j,ch] = 0
                elif value > 255:
                    Sharped_img[i, j,ch] = 255
                else:
                    Sharped_img[i, j,ch] = value
    Sharped_img = Image.fromarray(cv.cvtColor(Sharped_img, cv.COLOR_BGR2RGB))               
    return Sharped_img 



def reduce_gray_levels(img, gray_levels):
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
    # Calculate gap between gray levels
    gap = 256 // gray_levels
    
    # Generate array of colors
    colors = np.arange(0, 256, gap)
    
    # Convert image to float for calculations
    img = img.astype(np.float32)
    
    # Apply gray level reduction
    temp = img / gap
    index = np.floor(temp).astype(np.int32)
    new_img = colors[index]
    
    # Convert image back to uint8 for display
    new_img = new_img.astype(np.uint8)
    new_img = Image.fromarray(cv.cvtColor(new_img, cv.COLOR_BGR2RGB))   
    return new_img