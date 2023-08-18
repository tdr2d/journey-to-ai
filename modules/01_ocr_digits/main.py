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

(images, labels), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

def preprocessing(img):
    th = cv2.threshold(img, THRESHOLD, 255, cv2.THRESH_BINARY)[1]
    out = remove_noise(th)
    out = skeletonize(out)
    out = fill_shapes(out)
    out = skeletonize(out)
    return out

def combine(ref, out):
    for i in range(SAMPLE):
        yield np.concatenate((ref[i], out[i]))

if __name__ == "__main__":
    ref_images = images[0:SAMPLE]
    out = list(map(preprocessing, ref_images))
    show_images(list(combine(ref_images, out)), columns=15, rows=int(SAMPLE/15)+1, labels=labels[0:SAMPLE])