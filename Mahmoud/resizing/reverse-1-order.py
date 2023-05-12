import cv2 as cv
import numpy as np
import math

def reverse_1_order(img, output_size):
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
    return output.astype('uint8')

def reverse_1_order_builtin(img, output_size): # built in
    # resize_bilinear
    h, w = output_size
    # This implementation is shorter and likely faster because it takes advantage of optimized C code for the interpolation calculation.
    return cv.resize(img, (w, h), interpolation=cv.INTER_LINEAR)
