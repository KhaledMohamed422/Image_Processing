import cv2 as cv
import numpy as np

source = cv.imread('image/source (2).png')
template = cv.imread('image/template (2).png')

class ShapeError(Exception):pass

def Histogram_Equalization(img):
    
    feq =  np.zeros(256)    
    if len(img.shape) == 2 :
        size = img.shape
        new_img = np.zeros(size, np.uint8)    
        for r in range(size[0]):
            for c in range(size[1]):
                feq[img[r,c]] +=1 
                
        # Calculate the cumulative histogram           
        feq_cum = np.cumsum(feq)
        sum_pixels = feq_cum[-1]
        
        feq = ((feq_cum / float(sum_pixels)*255))
        for r in range(size[0]):
            for c in range(size[1]):
                new_img[r,c] =  feq[img[r,c]] 
        
    else:
        size = img.shape
        new_img = np.zeros(size, np.uint8)    
        for r in range(size[0]):
            for c in range(size[1]):
                for ch in range(size[2]):
                    feq[img[r,c,ch]] += 1 
                      
        # Calculate the cumulative histogram           
        feq_cum = np.cumsum(feq)
        sum_pixels = feq_cum[-1]
        feq = ((feq_cum / float(sum_pixels)*255))
        
        for r in range(size[0]):
            for c in range(size[1]):
                for ch in range(size[2]):
                   new_img[r,c,ch] =  round(feq[img[r,c,ch]]) 
    return new_img

def match_histograms(source, template):
    try:
        if len(source.shape) != len(template.shape):
            raise ShapeError("the shape of source and template not same")
        
        # Equalize the histograms of the images
        source_equalized = Histogram_Equalization(source)
        template_equalized = Histogram_Equalization(template)
        
        # Compute the histograms of the equalized images
        source_hist, _ = np.histogram(source_equalized, bins=256, range=(0, 255))
        template_hist, _ = np.histogram(template_equalized, bins=256, range=(0, 255))
        
        # Normalize the histograms
        source_hist = source_hist / float(source.size)
        template_hist = template_hist / float(template.size)
        
        # Compute the cumulative sum of the histograms
        source_cumsum = np.cumsum(source_hist)
        template_cumsum = np.cumsum(template_hist)
        
        # Normalize the cumulative sums
        source_cumsum = source_cumsum / source_cumsum[-1]
        template_cumsum = template_cumsum / template_cumsum[-1]
        
        # Create a lookup table for mapping pixel values
        lookup_table = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            j = 255
            while template_cumsum[j] > source_cumsum[i] and j > 0:
                j -= 1
            lookup_table[i] = j
        
        # Map the pixel values of the source image to the template image
        if len(source.shape) == 2 :
            result = np.zeros(source.shape, dtype=np.uint8)
            for i in range(source.shape[0]):
                for j in range(source.shape[1]):
                    result[i, j] = lookup_table[source_equalized[i, j]]
            
       # Map the pixel values of the source image to the template image             
        else: 
            result = np.zeros(source.shape, dtype=np.uint8)
            for i in range(source.shape[0]):
                for j in range(source.shape[1]):
                    for ch in range(source.shape[2]):
                        result[i, j , ch] = lookup_table[source_equalized[i, j, ch]]
        print(result)                
        return result.astype('uint8')
    
    
    except ShapeError:
        print(ShapeError)

    

Matched_Image = match_histograms(source,template)
cv.imshow('Matched Image', Matched_Image)
cv.imshow('source',source)
cv.imshow('template',template)
cv.waitKey(0)    
