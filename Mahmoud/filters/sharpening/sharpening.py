#########################################
## Name:  Mahmoud Ahmed Zaytoni     #####
## Id:    2021425                   #####
## Group: G5                        #####
## assignment:  3                   #####
#########################################
import cv2 as cv
import numpy as np

def apply_sharpening(img, mask):
    rows,cols,channels = img.shape
    output = img.copy()
    
    border_size = 3 // 2
    # Pad the input image with REPLICATE to handle borders
    padded_img = cv.copyMakeBorder(output, border_size, border_size, border_size, border_size,cv.BORDER_REPLICATE)
    
    for i in range(border_size, rows + border_size):
        for j in range(border_size, cols + border_size):
            pixel_value = 0
            for k in range(-1, 2):
                for l in range(-1, 2):
                    pixel_value += padded_img[i+k, j+l].astype('int16') * mask[k+1, l+1]
            output[i-border_size, j-border_size] = np.clip(pixel_value, 0, 255)
    return output.astype('uint8')

#org = cv.imread('org.png')
org = cv.imread('org copy.png')
# cv.imshow('original', org)
# cv.waitKey(0)

mask = np.array([[0, 1, 0],
                [1, -4, 1],
                [0, 1, 0]])

mask2 = np.array([[-1, -1, -1],
                [-1, 9, -1],
                [-1, -1, -1]])

# Line Sharpening
mask3 = np.array([[0, 1, 0],
                [0, 1, 0],
                [0, -1, 0]])

out = apply_sharpening(org, mask3)
# cv.imshow('modified', out)
# cv.waitKey(0)

import matplotlib.pyplot as plt


# Display the original and filtered images side by side
fig, axs = plt.subplots(1, 2)
axs[0].imshow(cv.cvtColor(org, cv.COLOR_BGR2RGB))
axs[0].set_title('Original Image')
axs[1].imshow(cv.cvtColor(out, cv.COLOR_BGR2RGB))
axs[1].set_title('Sharpining Image')
plt.show()
