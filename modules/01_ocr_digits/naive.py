import tensorflow as tf
import numpy as np
import tqdm
import cv2

THRESHOLD = 127
(images, labels), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
def show_images(img):
    cv2.imshow('tmp.png', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def count_horizontal(img, threshold=THRESHOLD):
    count = 0
    for i in range(28):
        if img[14][i] > threshold:
            count += 1
    return count
def count_vertical(img, threshold=THRESHOLD):
    count = 0
    for i in range(28):
        if img[i][14] > threshold:
            count += 1
    return count
def count_diag_tl_to_br(img, threshold=THRESHOLD):
    count = 0
    for i in range(28):
        if img[i][i] > threshold:
            count += 1
    return count
def count_diag_tr_to_bl(img, threshold=THRESHOLD):
    count = 0
    for i in range(28):
        if img[i][27-i] > threshold:
            count += 1
    return count

counts = list([{'h': {}, 'v': {}, 'diag_tl_to_br': {}, 'diag_tr_to_bl': {}} for i in range(10)])
h_counts, v_counts = np.array([], dtype=int), np.array([], dtype=int)
for i in tqdm.tqdm(range(len(images))):
    if labels[i] == 0:
        # print(images[i])
        h_counts = np.append(h_counts, count_horizontal(images[i]))
        if count_horizontal(images[i]) == 20:
            show_images(images[i])
        v_counts = np.append(v_counts, count_vertical(images[i]))
        # print(count_horizontal(images[i]))

print(counts[0])
counts[0]['h']['avg'] = np.average(h_counts)
counts[0]['h']['med'] = np.median(h_counts)
counts[0]['h']['min'] = np.min(h_counts)
counts[0]['h']['max'] = np.max(h_counts)

counts[0]['v']['avg'] = np.average(v_counts)
counts[0]['v']['med'] = np.median(v_counts)
counts[0]['v']['min'] = np.min(v_counts)
counts[0]['v']['max'] = np.max(v_counts)

print(counts[0])
# training = {0: {h: {min, max, med, avg}}, v: {}}

# def guess_number(pixel_mat):
