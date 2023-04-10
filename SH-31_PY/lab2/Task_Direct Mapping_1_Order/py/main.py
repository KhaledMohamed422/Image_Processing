import cv2 as cv
import numpy as np

img = cv.imread("1.png",0)
fact = int(input("Enter the facter resize :"))



def drict_1_order(img, fact):
    channels = 0

    if len(img.shape) == 3:
        # colored image
        height, width, channels = img.shape
        new_img = np.zeros((height * fact, width * fact, channels), np.uint8)

        # Add the old vals to the new image
        for i in range(height):
            for j in range(width):
                new_img[i * fact, j * fact, :] = img[i, j, :]

        # iterate for each row
        for i in range(channels):
            for row in new_img:
                # minimum value and index
                print(row[i])
                min_val = np.min(row[i])
                min_idx = np.argmin(row[i])

                # maximum value and index
                max_val = np.max(row[i])
                max_idx = np.argmax(row[i])

                # Pixel(i)= Round(((Max - Min)/Fact)*i + Min))
                new_val = round(((max_val - min_val) / fact) * i + min_val)

                min_idx = min(min_idx, max_idx)
                max_idx = max(min_idx, max_idx)

                row[i][min_idx:max_idx + 1] = new_val
                row[i][max_idx:] = max_val

        # iterate for each col
        for j in range(channels):
            for col in new_img.T:
                # minimum value and index

                min_val = np.min(col[j])
                min_idx = np.argmin(col[j])

                # maximum value and index
                max_val = np.max(col[j])
                max_idx = np.argmax(col[j])

                # Pixel(i)= Round(((Max - Min)/Fact)*i + Min))
                new_val = round(((max_val - min_val) / fact) * j + min_val)

                min_idx = min(min_idx, max_idx)
                max_idx = max(min_idx, max_idx)

                col[j][min_idx:max_idx + 1] = new_val
                col[j][max_idx:] = max_val

    else:
        # gray scal image

        height, width = img.shape
        new_img = np.zeros((height * fact, width * fact), np.uint8)

        # Add the old vals to the new image
        for i in range(height):
            for j in range(width):
                new_img[i * fact, j * fact] = img[i, j]
            print(new_img)
            # iterate for each row
            for row in new_img:
                # minimum value and index
                print(row)
                min_val = np.min(row)
                min_idx = np.argmin(row)

                # maximum value and index
                max_val = np.max(row)
                max_idx = np.argmax(row)

                # Pixel(i)= Round(((Max - Min)/Fact)*i + Min))
                new_val = round(((max_val - min_val) / fact) * i + min_val)

                min_idx = min(min_idx, max_idx)
                max_idx = max(min_idx, max_idx)

                row[min_idx:max_idx + 1] = new_val
                row[max_idx:] = max_val

            # iterate for each col
            for col in new_img.T:
                # minimum value and index

                min_val = np.min(col)
                min_idx = np.argmin(col)

                # maximum value and index
                max_val = np.max(col)
                max_idx = np.argmax(col)

                # Pixel(i)= Round(((Max - Min)/Fact)*i + Min))
                new_val = round(((max_val - min_val) / fact) * j + min_val)

                min_idx = min(min_idx, max_idx)
                max_idx = max(min_idx, max_idx)

                col[min_idx:max_idx + 1] = new_val
                col[max_idx:] = max_val
    cv.imshow('Orginal', img)
    cv.imshow('Resize', new_img)
    cv.waitKey(0)

drict_1_order(img,fact)









