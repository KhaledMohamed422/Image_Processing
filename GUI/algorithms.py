import cv2 as cv
import numpy as np
from math import floor
from PIL import Image
import matplotlib.pyplot as plt

######################Tab1############
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

def image_negative(img):
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
    img_out = img.copy()
    img_out = 255 - img
    img_out = Image.fromarray(cv.cvtColor(img_out, cv.COLOR_BGR2RGB))
    return img_out

def histogram_equalization(img):
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
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

def get_histogram(img):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])

    # Compute the cumulative distribution function
    cdf = hist.cumsum()
    return np.round((cdf / cdf.max()) * 255)

def histogram_matching(src_img, ref_img_path):
    src_img = cv.cvtColor(np.array(src_img), cv.COLOR_RGB2BGR)
    ref_img = cv.imread(ref_img_path)
    
    hist1 =  get_histogram(src_img)
    hist2 = get_histogram(ref_img)

    rows, cols = src_img.shape[:2]

    colors = np.zeros(256, dtype=np.uint8)

    for i in range(len(colors)):
        diffs = np.abs(hist1[i] - hist2)
        min_idx = np.argmin(diffs)
        colors[i] = min_idx
    
    for r in range(rows):
        for c in range(cols):
            src_img[r,c] = colors[src_img[r,c]]

    src_img = Image.fromarray(cv.cvtColor(src_img, cv.COLOR_BGR2RGB))
    return src_img

def Contrast(img,new_min,new_max):

    if isinstance(img, Image.Image):
        img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)

    if new_min < 0 or new_max > 255:
        return img
    
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
              
    new_img = Image.fromarray(cv.cvtColor(new_img, cv.COLOR_BGR2RGB))       
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

def add_image(img1, img2_path):  
    img2 = cv.imread(img2_path)
    img1 = cv.cvtColor(np.array(img1), cv.COLOR_RGB2BGR)
    

    img2 = cv.resize(img2, (img1.shape[1], img1.shape[0])) # resize img2 to match img1

    NewImage = np.zeros_like(img1)
    NewImage = img1 + img2

    NewImage = Image.fromarray(cv.cvtColor(np.array(NewImage.clip(0, 255).astype(np.uint8)), cv.COLOR_BGR2RGB))            
    return NewImage          

def sub_image(img1, img2_path):
    img2 = cv.imread(img2_path)
    img1 = cv.cvtColor(np.array(img1), cv.COLOR_RGB2BGR)

    img2 = cv.resize(img2, (img1.shape[1], img1.shape[0])) # resize img2 to match img1

    NewImage = np.zeros_like(img1)
    NewImage = img1 - img2

    NewImage = Image.fromarray(cv.cvtColor(np.array(NewImage.clip(0, 255).astype(np.uint8)), cv.COLOR_BGR2RGB))              
    return NewImage



      
def Drawing_the_histogram(img, name='img', title='Histogram'):
    # Convert the image to the appropriate type
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
    img = np.array(img)

    # Compute the histogram of the image
    hist = cv.calcHist([img], [0], None, [256], [0, 256])

    # Resize the image
    img_resized = cv.resize(img, (256, 256))

    # Plot the image and the histogram side by side
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))

    # Plot the image in the first row
    axs[0].imshow(cv.cvtColor(img_resized, cv.COLOR_BGR2RGB))
    axs[0].axis('off')

    # Plot the histogram in the second row
    axs[1].plot(hist, color='black')
    axs[1].bar(np.arange(len(hist)), hist.ravel(), color='black')
    axs[1].set_ylabel('Number of Pixels')
    axs[1].set_xlabel('Pixel Value')

    # Save the figure and return it as an array
    fig.savefig("hist_" + name)
    fig.show() 
    return img




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

def ideal_lowpass_filter(img, D0):
  img = cv.cvtColor(np.array(img), cv.COLOR_RGB2GRAY)
#   img = cv.cvtColor(np.array(img), cv.COLOR_BGR2GRAY)

  fft = np.fft.fft2(img)
  fft_image_shifted = np.fft.fftshift(fft)

  real = np.real(fft_image_shifted)
  imaginary = np.imag(fft_image_shifted)

  m, n = img.shape[:2]
  center = (m // 2, n // 2)
  filter = np.zeros((m, n), dtype=np.float32)
  for i in range(m):
    for j in range(n):
      D = np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2)
      if D <= D0:
        filter[i,j] = 1

  real = real * filter
  imaginary = imaginary * filter

  # revers transform to get image
  filtered_fft_image = np.fft.ifftshift(real + 1j * imaginary)

  filterd_image = np.abs(np.fft.ifft2(filtered_fft_image))
  filterd_image = filterd_image / filterd_image.max() * 255 # normalize image
  filterd_image = filterd_image.astype(np.uint8)
  img = Image.fromarray(cv.cvtColor(filterd_image, cv.COLOR_GRAY2RGB))  
  return img


########Tab4-lab7########
def ideal_highpass_filter(img,d):
    
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2GRAY)

    # img = cv.cvtColor(np.array(img), cv.COLOR_BGR2GRAY)  
    img_f = np.fft.fft2(img)
    img_fsh = np.fft.fftshift(img_f)
    
    
    
    img_fsh_real  = np.real(img_fsh)
    img_fsh_imag = np.imag(img_fsh)
    rows, cols = img_f.shape

    dist = np.zeros((rows, cols), dtype=np.float32)
   
    for i in range(rows):
        for c in range(cols):
            dist[i,c] = np.sqrt( (i-rows/2)**2 + (c-cols/2)**2 )

    
    mask = np.ones((rows, cols))
    mask[dist <= d] = 0
    
        
  
    img_fsh_real = img_fsh_real * mask

    img_fsh_imag = img_fsh_imag * mask

    
    img_fsh = np.fft.ifftshift(img_fsh_real + 1j * img_fsh_imag )

    filterd_image = np.abs(np.fft.ifft2(img_fsh))
    filterd_image = filterd_image / filterd_image.max() * 255 

    img = Image.fromarray(cv.cvtColor(filterd_image.astype(np.uint8), cv.COLOR_GRAY2RGB))  
    return img



def Butterworth_lowpass_filter(img,d,n=2):
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2GRAY)

    # img = cv.cvtColor(np.array(img), cv.COLOR_BGR2GRAY) 


    img_f = np.fft.fft2(img)

    img_fsh = np.fft.fftshift(img_f)
    
    
    img_fsh_real  = np.real(img_fsh)
    img_fsh_imag = np.imag(img_fsh)

    rows, cols  = img_f.shape
   
    
    dist = np.zeros((rows, cols), dtype=np.float32)
   
    for i in range(rows):
        for c in range(cols):
            dist[i,c] = np.sqrt( (i-rows/2)**2 + (c-cols/2)**2 )       

    
    mask =  1 / (1 + (dist/d)**(2*n))
 
              
  
    img_fsh_real = img_fsh_real * mask

    img_fsh_imag = img_fsh_imag * mask

    
    img_fsh = np.fft.ifftshift(img_fsh_real + 1j * img_fsh_imag )

    filterd_image = np.abs(np.fft.ifft2(img_fsh))
    filterd_image = filterd_image / filterd_image.max() * 255 

    img = Image.fromarray(cv.cvtColor(filterd_image.astype(np.uint8), cv.COLOR_GRAY2RGB))  
    return img


def Butterworth_High_Pass_Filter(img,d,n=2):
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2GRAY)
    # img = cv.cvtColor(np.array(img), cv.COLOR_BGR2GRAY)

    img_f = np.fft.fft2(img)

    img_fsh = np.fft.fftshift(img_f)
    
    
    
    img_fsh_real  = np.real(img_fsh)
    img_fsh_imag = np.imag(img_fsh)

    rows, cols  = img_f.shape
    
    dist = np.zeros((rows, cols), dtype=np.float32)
   
    for i in range(rows):
        for c in range(cols):
            dist[i,c] = np.sqrt( (i-rows/2)**2 + (c-cols/2)**2 )       

    
    mask = 1 -  (1 / (1 + (dist/d)**(2*n)) )
 
              
  
    img_fsh_real = img_fsh_real * mask

    img_fsh_imag = img_fsh_imag * mask

    
    img_fsh = np.fft.ifftshift(img_fsh_real + 1j * img_fsh_imag )

    filterd_image = np.abs(np.fft.ifft2(img_fsh))
    filterd_image = filterd_image / filterd_image.max() * 255 

    img = Image.fromarray(cv.cvtColor(filterd_image.astype(np.uint8), cv.COLOR_GRAY2RGB))  
    return img

def gaussian_lowpass_filter(img,d):
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2GRAY)
    # img = cv.cvtColor(np.array(img), cv.COLOR_BGR2GRAY)
    img_f = np.fft.fft2(img)

    img_fsh = np.fft.fftshift(img_f)
    
    rows, cols  = img_f.shape
    
    img_fsh_real  = np.real(img_fsh)
    img_fsh_imag = np.imag(img_fsh)

   
    dist = np.zeros((rows, cols), dtype=np.float32)
   
    for i in range(rows):
        for c in range(cols):
            dist[i,c] = np.sqrt( (i-rows/2)**2 + (c-cols/2)**2 )       

    
    mask = np.exp(-(dist**2) / (2*(d**2)))
 
              
  
    img_fsh_real = img_fsh_real * mask

    img_fsh_imag = img_fsh_imag * mask

    
    img_fsh = np.fft.ifftshift(img_fsh_real + 1j * img_fsh_imag )

    filterd_image = np.abs(np.fft.ifft2(img_fsh))
    filterd_image = filterd_image / filterd_image.max() * 255 

    img = Image.fromarray(cv.cvtColor(filterd_image.astype(np.uint8), cv.COLOR_GRAY2RGB))  
    return img


def Gaussian_High_Pass_Filter (img,d):
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2GRAY)
    # img = cv.cvtColor(np.array(img), cv.COLOR_BGR2GRAY)
    img_f = np.fft.fft2(img)

    img_fsh = np.fft.fftshift(img_f)
    
    rows, cols  = img_f.shape
    
    img_fsh_real  = np.real(img_fsh)
    img_fsh_imag = np.imag(img_fsh)

   
    dist = np.zeros((rows, cols), dtype=np.float32)
   
    for i in range(rows):
        for c in range(cols):
            dist[i,c] = np.sqrt( (i-rows/2)**2 + (c-cols/2)**2 )       

    
    mask = 1 - np.exp(-(dist**2) / (2*(d**2)))
 
              
  
    img_fsh_real = img_fsh_real * mask

    img_fsh_imag = img_fsh_imag * mask

    
    img_fsh = np.fft.ifftshift(img_fsh_real + 1j * img_fsh_imag )

    filterd_image = np.abs(np.fft.ifft2(img_fsh))
    filterd_image = filterd_image / filterd_image.max() * 255 

    img = Image.fromarray(cv.cvtColor(filterd_image.astype(np.uint8), cv.COLOR_GRAY2RGB))  
    return img

########Export Tab########
import math

def convert_to_bgr(image):
    if len(image.shape) == 2:
        # Grayscale image
        bgr_image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    elif len(image.shape) == 3 and image.shape[2] == 3:
        # RGB image
        bgr_image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    else:
        raise ValueError(f"Unsupported image format {image.shape}, {image.shape[2]}")

    return bgr_image

def reverse_1_order(img, output_size):
    img = np.array(img)
    img = convert_to_bgr(img)
    print(img.shape)
    row_ratio = img.shape[0] / output_size[0]
    col_ratio = img.shape[1] / output_size[1]

    output = np.zeros((output_size[0], output_size[1], img.shape[2]))

    for channel in range(img.shape[2]):

        for new_x in range(output_size[0]):
            old_x = new_x * row_ratio
            x1 = math.floor(old_x) # we can use int()
            x2 = x1 + 1

            if x2 >= img.shape[0]:
                x2 = x1
            x_fraction = abs(old_x-x1)

            for new_y in range(output_size[1]):
                old_y = new_y * col_ratio
                y1 = math.floor(old_y)
                y2 = y1 + 1

                if y2 >= img.shape[1]:
                    y2 = y1
                y_fraction = abs(old_y-y1)

                p1 = img[x1, y1, channel]
                p2 = img[x2, y1, channel]
                p3 = img[x1, y2, channel]
                p4 = img[x2, y2, channel]

                z1 = p1 * (1 - x_fraction) + p2*(x_fraction)
                z2 = p3 * (1 - y_fraction) + p4*(y_fraction)
                new_pixel = z1 * (1 - y_fraction) + z2 * (y_fraction)

                output[new_x, new_y, channel] = math.floor(new_pixel)

    output = Image.fromarray(cv.cvtColor(output.astype(np.uint8), cv.COLOR_BGR2RGB))  
    return output
