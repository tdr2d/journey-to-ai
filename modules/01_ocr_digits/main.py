import ssl
import tensorflow as tf
import numpy as np
import tqdm
import cv2
from utils import *
import math
ssl._create_default_https_context = ssl._create_unverified_context

THRESHOLD = 127
SAMPLE = 100


def preprocessing(img):
    if img.ndim == 3:
        th = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), THRESHOLD, 255, cv2.THRESH_BINARY_INV)[1]
    else:
        th = cv2.threshold(img, THRESHOLD, 255, cv2.THRESH_BINARY)[1]

    out = remove_noise(th)
    out = skeletonize(out)
    out = fill_shapes(out)
    out = skeletonize(out)

    out = cv2.bitwise_not(out)
    out_color = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    contours, hierarchy = cv2.findContours(out, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    for c in contours:
        cv2.drawContours(out_color, contours, -1, (0,0,255), 1)

        # for eps in np.linspace(0.001, 0.05, 10):
        #     peri = cv2.arcLength(c, True)
        #     approx = cv2.approxPolyDP(c, eps * peri, True)
        #     if len(approx == 4):
        #         cv2.drawContours(out_color, [approx], -1, (0,0,255), 1)
        #         break
    
    # img = cv2.resize(img, (28, 28))

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