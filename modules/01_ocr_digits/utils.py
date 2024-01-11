import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_images(images, columns=2, rows=2, labels=[], show=True):
    fig = plt.figure(figsize=(15, 8))
    i = 1
    for img in images:
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
        plt.axis('off')
        if len(labels):
            plt.title(f"{labels[i-1]}", loc='left', fontsize=8, pad=2)
        i += 1
    if show:
        fig.tight_layout()
        plt.show()

def cv2show(img):
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def remove_noise(img):
    kernel = np.ones((1,1),np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def fill_shapes(img):
    fill1 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_CROSS, (2,2)))
    fill2 = cv2.morphologyEx(fill1, cv2.MORPH_CLOSE, np.ones((2,2),np.uint8))
    return fill2

def skeletonize(img):
    img = img.copy()
    skel = img.copy()
    skel[:,:] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

    while True:
        eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
        temp  = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img[:,:] = eroded[:,:]
        if cv2.countNonZero(img) == 0:
            break
    return skel