import cv2 as cv
import numpy as np



img1 = cv.imread('image/source.png')

img2= cv.imread('image/template.png')

def Image_Negative(img):
    img = 255 - img          
    return img

def Quantization(img, bit_size=8):
    gray_level = 2**bit_size
    gap = 256 / gray_level
    colors = np.arange(gap, 256, gap)
    index = np.floor(img / gap).astype(int)
    new_img = colors[index]
    return new_img.astype('uint8')

def add_image(source_img, op_img):
    try:
       
        height, width, channels = source_img.shape
        result_add = np.zeros((height, width, channels), dtype='uint8')
       
        op_img = cv.resize(op_img, (source_img.shape[1], source_img.shape[0]))
        
        
        for y in range(height):
            for x in range(width):
                for c in range(channels):             
                    result_add[y, x, c] = source_img[y, x, c].astype(int) + op_img[y, x, c].astype(int)
                    
        return result_add.astype('uint8')            
    except ValueError:
        print('Error: Image sizes do not match.')
        return None

def sub_image(source_img, op_img):
    try:
        height, width, channels = source_img.shape        
        result_subtract = np.zeros((height, width, channels), dtype='uint8')
        
        op_img = cv.resize(op_img, (source_img.shape[1], source_img.shape[0]))
     
        for y in range(height):
            for x in range(width):
                for c in range(channels):                       
                    result_subtract[y, x, c] = source_img[y, x, c].astype(int) - op_img[y, x, c].astype(int)  
                    
        return result_subtract.astype('uint8')
        
    except ValueError:
        print('Error: Image sizes do not match.')
        return None



cv.imshow('soure_img',img1)
cv.imshow('op_img',img2)
cv.imshow('add_image',add_image(img1,img2)) 

try:
    cv.imshow('add_image',add_image(img1,img2)) 
    cv.imshow('sub_image',sub_image(img1, img2)) 
  
finally:    
    cv.waitKey(0)                   