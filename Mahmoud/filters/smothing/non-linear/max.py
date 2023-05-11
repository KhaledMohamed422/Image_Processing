#########################################
## Name:  Mahmoud Ahmed Zaytoni     #####
## Id:    2021425                   #####
## Group: G5                        #####
## assignment:  3                   #####
#########################################
import cv2 as cv
import numpy as np

def apply_median_filter(img, mask_size):
    rows, cols, channels = img.shape
    output = np.zeros((rows, cols, channels), dtype=np.uint8)
    
    border_size = mask_size // 2
    
    # Pad the input image with REPLICATE to handle borders
    padded_img = cv.copyMakeBorder(img, border_size, border_size, border_size, border_size,cv.BORDER_REPLICATE)
    
    for i in range(border_size, rows + border_size):
        for j in range(border_size, cols + border_size):
            # Get the neighborhood of the current pixel
            neighborhood = padded_img[i-border_size:i+border_size+1, j-border_size:j+border_size+1]
            # Calculate the median of the neighborhood along the third axis (channels)
            median = np.median(neighborhood, axis=(0,1))
            #print(median)
            output[i - border_size, j - border_size] = median
    return output

def apply_max_filter(img, mask_size):
    rows, cols, channels = img.shape
    output = np.zeros((rows, cols, channels), dtype=np.uint8)
    
    border_size = mask_size // 2
    
    # Pad the input image with REPLICATE to handle borders
    padded_img = cv.copyMakeBorder(img, border_size, border_size, border_size, border_size,cv.BORDER_REPLICATE)
    
    for i in range(border_size, rows + border_size):
        for j in range(border_size, cols + border_size):
            # Get the neighborhood of the current pixel
            neighborhood = padded_img[i-border_size:i+border_size+1, j-border_size:j+border_size+1]
            
            # Calculate the max of the neighborhood along the third axis (channels)
            mx = np.max(neighborhood, axis=(0,1))
            output[i - border_size, j - border_size] = mx
    return output

def apply_min_filter(img, mask_size):
    rows, cols, channels = img.shape
    output = np.zeros((rows, cols, channels), dtype=np.uint8)
    
    border_size = mask_size // 2
    
    # Pad the input image with REPLICATE to handle borders
    padded_img = cv.copyMakeBorder(img, border_size, border_size, border_size, border_size,cv.BORDER_REPLICATE)
    
    for i in range(border_size, rows + border_size):
        for j in range(border_size, cols + border_size):
            # Get the neighborhood of the current pixel
            neighborhood = padded_img[i-border_size:i+border_size+1, j-border_size:j+border_size+1]
            
            # Calculate the Min of the neighborhood along the third axis (channels)
            mn = np.min(neighborhood, axis=(0,1))
            output[i - border_size, j - border_size] = mn
    return output
org = cv.imread('org.jpg')

import matplotlib.pyplot as plt
# create a 2x2 subplot grid
fig, axs = plt.subplots(2, 2)

# plot the original image
axs[0, 0].imshow(org, cmap='gray')
axs[0, 0].set_title('Original')

# plot the filtered images
axs[0, 1].imshow(apply_min_filter(org, 3), cmap='gray')
axs[0, 1].set_title('min filter with mask 3x3')

axs[1, 0].imshow(apply_max_filter(org, 5), cmap='gray')
axs[1, 0].set_title('max filter with mask 5x5')

axs[1, 1].imshow(apply_median_filter(org, 5), cmap='gray')
axs[1, 1].set_title('median filter with mask 5x5')

fig.suptitle('Non Linear Smothing filters')
# display the plot
plt.show()