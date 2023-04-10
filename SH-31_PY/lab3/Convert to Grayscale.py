import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread('image/1.png')

def rgp_2_gray_Single_color_channel(img,i=0):
    if i<=2 and i>=0:
        return img[:,:,i].astype('uint8')

def rgp_2_gray_Averaging(img):
    gray_img =  ((img[:,:,0]+img[:,:,1]+img[:,:,2])/3)
    gray_img.round
    return gray_img.astype('uint8')

def rgp_2_gray_Luminance(img):
    gray_img = ( (0.3* (img[:,:,0])) + (0.59*(img[:,:,1])) + (0.11*(img[:,:,2])) )
    
    return gray_img.astype('uint8')

def rgp_2_gray_Desaturation_avg(img):
    red_channel = img[:, :, 0]
    green_channel = img[:, :, 1]
    blue_channel = img[:, :, 2]

    max_rgb = np.maximum.reduce([red_channel, green_channel, blue_channel])
    min_rgb = np.minimum.reduce([red_channel, green_channel, blue_channel])

    gray_image = (max_rgb + min_rgb) / 2
    return gray_image.astype('uint8')

def rgp_2gray_Decomposing(img,i=0):
    red_channel = img[:, :, 0]
    green_channel = img[:, :, 1]
    blue_channel = img[:, :, 2]
    if i==0:
        gray_image = np.maximum.reduce([red_channel, green_channel, blue_channel])
        return gray_image.astype('uint8')
    else:
        gray_image = np.minimum.reduce([red_channel, green_channel, blue_channel])
        return gray_image.astype('uint8')
    
def Drawing_the_histogram(img,name,title):
    _histrogram = cv.calcHist([img],[0],None,[256],[0,256])
    plt.figure()
    plt.title(title)
    plt.plot(_histrogram, color='black')
    plt.bar(np.arange(len(_histrogram)), _histrogram, color='black')
    plt.ylabel('Number of Pixels')
    plt.xlabel('Pixel Value')
    plt.savefig("hist_" + name)




cv.imshow("rgp imagr",img)
cv.imshow('gray Single color channel 0',rgp_2_gray_Single_color_channel(img,0))
cv.imshow('gray Single color channel 1',rgp_2_gray_Single_color_channel(img,1))
cv.imshow('gray Single color channel 2',rgp_2_gray_Single_color_channel(img,2))
cv.imshow('Averaging ',rgp_2_gray_Averaging(img))
cv.imshow('Luminance ',rgp_2_gray_Luminance(img))
cv.imshow('Desaturation avg val',rgp_2_gray_Desaturation_avg(img)) 
cv.imshow('Desaturation max val',rgp_2gray_Decomposing(img,0))   
cv.imshow('Desaturation max val',rgp_2gray_Decomposing(img,0))   
cv.imshow('Desaturation min val',rgp_2gray_Decomposing(img,1)) 
cv.imshow("rgp imagr",img) 
cv.waitKey(0)  
Drawing_the_histogram(img,'hostgram','img1')   
    
       
