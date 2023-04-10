import cv2 as cv
import numpy as np


def apply_DM_1(arr, Fact):
    # take the orgain idc
    li = np.arange(0,len(arr),Fact)

    # loop for each max , min in the orgain array

    for i in range(1,len(li)):
        start = li[i-1] # max 
        end = li[i] # min
        slope = (arr[end] - arr[start])
        for i in range(start+1, end):
            arr[i] = round( (slope/Fact) * ( i - start ) )+ arr[start]
            
    if int(len(li)) % 2 == 1:
        last_pixel = li[-1]
        arr[last_pixel:] = [arr[last_pixel]]*len(arr[last_pixel:])
    return arr


def direct_1_order_gray(img, factor):
    # Get the dimensions of the input image
    height, width = img.shape

    # Compute the new dimensions of the resized image
    new_height, new_width = height * factor, width * factor

    # Create a new image of zeros with the new dimensions
    new_img = np.zeros((new_height, new_width))

    # Move the pixels from the old image to the new image
    for i in range(height):
        for j in range(width):
            new_img[i * factor, j * factor] = img[i, j]

    # Iterate over each row in the new image and interpolate the values between consecutive zeros
    for i in range(new_height):
        new_img[i] = apply_DM_1(new_img[i], factor)

    # Transpose the new image to iterate over the columns
    new_img_T = new_img.T

    # Iterate over each column in the new image and interpolate the values between consecutive zeros
    for i in range(new_width):
        new_img_T[i] = apply_DM_1(new_img_T[i], factor)

    # Transpose the new image back to its original orientation
    new_img = new_img_T.T

    return new_img.astype('uint8') 


def direct_1_order_rgb(img, factor):
    # Get the dimensions of the input image
    height, width, channels = img.shape

    # Compute the new dimensions of the resized image
    new_height, new_width = height * factor, width * factor

    # Create a new image of zeros with the new dimensions
    new_img = np.zeros((new_height, new_width, channels))

    # Move the pixels from the old image to the new image
    for i in range(height):
        for j in range(width):
            new_img[i * factor, j * factor, :] = img[i, j, :]

    # Iterate over each channel in the new image
    for channel in range(channels):
        # Iterate over each row in the new image
        for i in range(new_height):
            new_img[i, :, channel] = apply_DM_1(new_img[i, :, channel], factor)

        # Iterate over each column in the new image
        for j in range(new_width):
            new_img[:, j, channel] = apply_DM_1(new_img[:, j, channel], factor)

    return new_img.astype('uint8') 



img=cv.imread('image/1.jpeg',0)
factor = 3
new_img = direct_1_order_gray(img, factor)
print(new_img)
cv.imshow('Orginal image',img)
cv.imshow(f'Resize image X{factor}',new_img)
cv.waitKey(0)
