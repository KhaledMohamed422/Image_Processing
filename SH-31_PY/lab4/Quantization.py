import cv2 as cv
import numpy as np

img = cv.imread('img/images.jpg')

def reduce_gray_levels(img, gray_levels):
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
    
    return new_img

# Load an example image
img = cv.imread('img/images.jpg')

# Reduce gray levels to 4
new_img = reduce_gray_levels(img, 2)

# Display the original and reduced images
cv.imshow('Original Image', img)
cv.imshow('Reduced Image', new_img)
cv.waitKey(0)
cv.destroyAllWindows()
