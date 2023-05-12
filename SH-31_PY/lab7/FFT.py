import numpy as np
import cv2 as cv


img = cv.imread('img/1.jpg',0)


# convert img form  Spectral to frquency domain.  
img_f = np.fft.fft2(img)

img_fsh = np.fft.fftshift(img_f)

img_fsh_real  = np.real(img_fsh)

img_fsh_imag = np.imag(img_fsh)




############################################
# reverse
img_fsh = np.fft.ifftshift(img_fsh_real + 1j * img_fsh_imag )

img = np.fft.ifft2(img_fsh)