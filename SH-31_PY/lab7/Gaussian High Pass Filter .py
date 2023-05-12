import numpy as np
import cv2 as cv


img = cv.imread('img/1.jpg',0)




def Gaussian_High_Pass_Filter (img,d):

    img_f = np.fft.fft2(img)

    img_fsh = np.fft.fftshift(img_f)
    
    rows, cols  = img_f.shape
    
    img_fsh_real  = np.real(img_fsh)
    img_fsh_imag = np.imag(img_fsh)

   
    dist = np.zeros((rows, cols))
   
    for i in range(rows):
        for c in range(cols):
            dist[i,c] = np.sqrt( (i-rows/2)**2 + (c-cols/2)**2 )       

    
    mask = 1 - np.exp(-(dist**2) / (2*(d**2)))
 
              
  
    img_fsh_real = img_fsh_real * mask

    img_fsh_imag = img_fsh_imag * mask

    
    img_fsh = np.fft.ifftshift(img_fsh_real + 1j * img_fsh_imag )

    img = np.fft.ifft2(img_fsh)
    
    img = np.uint8(np.abs(img))
    return img

cv.imshow('orgin',img)
cv.imshow('Gaussian_High_Pass_Filter effect',Gaussian_High_Pass_Filter(img, 15))
cv.waitKey(0)





