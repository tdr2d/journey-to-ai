import cv2
import numpy as np
import logging


# https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html
# https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html
# https://maker.pro/raspberry-pi/tutorial/grid-detection-with-opencv-on-raspberry-pi
# https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_begin/py_contours_begin.html
# https://stackoverflow.com/questions/14997733/advanced-square-detection-with-connected-region


data = [
    'datasets/sudoku/sudoku-screen.png',
    'datasets/sudoku/sudoku-screen-2.png',
    'datasets/sudoku/sudoku-screen-3.png',
    'datasets/sudoku/sudoku-screen-4.jpg'
]

img =  cv2.imread(data[0])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # gray filter
blur = cv2.bilateralFilter(gray, 3, 75, 75) # use bilateral filter in order not to blur edges

# denoise
kernel = np.ones((3,3), np.uint8)
blur = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel) 

# Filter
th1 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
th2 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
laplacian = cv2.Laplacian(blur,cv2.CV_64F)
canny = cv2.Canny(blur, 60, 180)

def filter_contours(c):
    _,_,w,h = cv2.boundingRect(c)
    ratio = min(w,h) / max(w,h)
    return w > 35 and h > 35 and cv2.isContourConvex(c) and ratio > 0.9


def extract_from_contour(img, contour, output_file=None):
    mask = np.zeros((img.shape),np.uint8)
    cv2.drawContours(mask,[contour],0, color=(255, 255, 255), thickness=cv2.FILLED)
    out = np.zeros_like(img)
    out[mask == 255] = img[mask == 255]
    if output_file:
        cv2.imwrite(output_file, out)
    else:
        return out

def nested_contours(images, contour):
    contours = []
    for img in images:
        extract = extract_from_contour(img, contour)
        contours += list(filter(filter_contours, cv2.findContours(extract, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]))

    return contours

def find_biggest_contour(images):
    contours = []
    for img in images:
        contours += list(filter(filter_contours, cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]))

    if len(contours):
        return sorted(contours, key=cv2.contourArea, reverse=True)[0]
    return None

biggest_contour = find_biggest_contour([th1, th2])
cv2.drawContours(img,[biggest_contour],0, color=(255, 0, 0), thickness=2)

nested = nested_contours([th1, th2, canny], biggest_contour)
cv2.drawContours(img,nested,-1, color=(0, 255, 0), thickness=1)


cv2.imwrite('data/tmp/00_base.png', img)
cv2.imwrite('data/tmp/01_gray.png', gray)
cv2.imwrite('data/tmp/02_blur.png', blur)
cv2.imwrite('data/tmp/03_th1.png', th1)
cv2.imwrite('data/tmp/03_th2.png', th2)
cv2.imwrite('data/tmp/04_laplacian.png', laplacian)
cv2.imwrite('data/tmp/05_canny.png', canny)
if biggest_contour is not None:
    extract_from_contour(img, biggest_contour, 'data/tmp/99_extract.png')
else:
    logging.error("No squared contour found")
