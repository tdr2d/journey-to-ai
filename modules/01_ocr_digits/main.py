import ssl
import tensorflow as tf
import numpy as np
import tqdm
import cv2
from utils import *
from rectangle import Rectangle
import math
ssl._create_default_https_context = ssl._create_unverified_context

THRESHOLD = 127 # optimizing threshold https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/OWENS/LECT2/node3.html
SAMPLE = 100
PIXELS_X = 28
PIXELS_Y = 28
MIN_CONTOUR_THRESHOLD = 0.05


def find_digit_contour(gray_img):
    # RETR_EXTERNAL child contour are ignored
    # CHAIN_APPROX_NONE return closest geometry fit
    contours = cv2.findContours(gray_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]  
    min_contour_height = int(img.shape[0] * MIN_CONTOUR_THRESHOLD)
    min_contour_width = int(img.shape[1] * MIN_CONTOUR_THRESHOLD)
    contour_quandidates = [] 
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if w > min_contour_width and w < img.shape[1] - 1 and h > min_contour_height and h < img.shape[0] - 1:
            contour_quandidates.append(Rectangle(x,y,w,h))

    if len(contour_quandidates):
        big_contour = contour_quandidates.pop()
        for r in contour_quandidates:
            big_contour = big_contour.merge(r)
        return big_contour
    return None


def center_in_background(background_img, img):
    assert background_img.shape[0] > img.shape[0]
    assert background_img.shape[1] > img.shape[1]

    ## (1) Find centers
    pt1 = int(background_img.shape[1] / 2), int(background_img.shape[0] / 2)
    pt2 = int(img.shape[1] / 2), int(img.shape[0] / 2)

    ## (2) Calc offset
    dx = pt1[0] - pt2[0]
    dy = pt1[1] - pt2[1]

    ## (3) do slice-op `paste`
    h,w = img.shape[:2]

    dst = background_img.copy()
    dst[dy:dy+h, dx:dx+w] = img
    return dst

def preprocessing(img):
    if img.ndim == 3:
        th = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), THRESHOLD, 255, cv2.THRESH_BINARY_INV)[1]
    else:
        th = cv2.threshold(img, THRESHOLD, 255, cv2.THRESH_BINARY)[1]

    out = remove_noise(th)
    out = skeletonize(out)
    out = fill_shapes(out)
    out = skeletonize(out)
    # out = cv2.bitwise_not(out)

    out_color = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    digit_contour = find_digit_contour(out)
    if digit_contour is not None:
        digit_contour.debug(out_color)
        
        # sub_image = digit_contour.get_sub_image(out)
        # mask_width = max(digit_contour.w, digit_contour.h) + 3
        # mask = np.zeros((mask_width, mask_width), np.uint8)

        # out_color = cv2.resize(sub_image, (PIXELS_X, PIXELS_Y))
        # mask = center_in_background(mask, sub_image)

        # todo rotate
        # mask = cv2.resize(mask, (PIXELS_X, PIXELS_Y))
    
    return out_color

def combine(ref, out):
    for i in range(SAMPLE):
        yield np.concatenate((ref[i], out[i]))

if __name__ == "__main__":
    img = cv2.imread('test/8_uncentered_unaligned.png')
    img = preprocessing(img)
    # cv2show(img)
    cv2.imwrite('tmp/test.png', img)
    
    # (images, labels), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # ref_images = images[0:SAMPLE]
    # out = list(map(preprocessing, ref_images))
    # out = list(combine(ref_images, out)) + [img]
    # show_images(out, columns=15, rows=int(SAMPLE/15)+2, labels=list(labels[0:SAMPLE]) + [8])