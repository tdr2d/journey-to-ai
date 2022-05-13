import cv2
import numpy as np
import logging
from utils.transform import four_point_transform

# https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html
# https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html
# https://maker.pro/raspberry-pi/tutorial/grid-detection-with-opencv-on-raspberry-pi
# https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_begin/py_contours_begin.html
# https://stackoverflow.com/questions/14997733/advanced-square-detection-with-connected-region


data = [
    'datasets/00_sudoku/00_base.png',
    'datasets/00_sudoku/01_small_blurred.png',
    'datasets/00_sudoku/02_newspaper.jpg'
]

class SudokuDetector:
    MIN_SIZE_CELL_DIVIDER = 20

    def __init__(self, from_image) -> None:
        self.img =  cv2.imread(from_image)
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY) # gray filter
        self.blur = cv2.bilateralFilter(self.gray, 4, 75, 75) # use bilateral filter in order not to blur edges

        # Filter
        self.th1 = cv2.bitwise_not(cv2.adaptiveThreshold(self.blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2))

        # Contours
        self.min_w_cell = self.img.shape[2] / self.MIN_SIZE_CELL_DIVIDER
        self.min_h_cell = self.img.shape[1] / self.MIN_SIZE_CELL_DIVIDER
        self.biggest_contour = self.find_biggest_contour(self.th1)


        if self.biggest_contour is None:
            logging.error("Sudoku not found")
        else:
            # extract = self.extract_from_contour(self.th1, self.biggest_contour, 'data/tmp/98_extract.png')
            # extract_img = self.extract_from_contour(self.th1, self.biggest_contour, 'data/tmp/98_extract.png')

            top_down = four_point_transform(self.th1, self.biggest_contour.reshape(4,2))
            top_down_img = four_point_transform(self.img, self.biggest_contour.reshape(4,2))
            nested_contours = self.find_nested_contours(top_down)
            # cv2.drawContours(self.img,[self.biggest_contour],0, color=(255, 0, 0), thickness=2)
            cv2.drawContours(top_down_img, nested_contours, -1, color=(0, 255, 0), thickness=1)
            print(f"Found {len(nested_contours)} nested contours")
            cv2.imwrite('data/tmp/98_top_down_img.png', top_down_img)
            cv2.imwrite('data/tmp/99_top_down.png', top_down)

        cv2.imwrite('data/tmp/00_base.png', self.img)
        cv2.imwrite('data/tmp/01_gray.png', self.gray)
        cv2.imwrite('data/tmp/02_blur.png', self.blur)
        cv2.imwrite('data/tmp/03_th1.png', self.th1)

    def filter_contours(self, c):
        _,_,w,h = cv2.boundingRect(c)
        ratio = min(w,h) / max(w,h)
        return w > self.min_w_cell and h >self.min_h_cell and cv2.isContourConvex(c) and ratio >= 0.9

    @staticmethod
    def extract_from_contour(img, contour, output_file=None):
        if contour is None:
            return img
        mask = np.zeros((img.shape),np.uint8)
        cv2.drawContours(mask,[contour],0, color=(255, 255, 255), thickness=cv2.FILLED)
        out = np.zeros_like(img)
        out[mask == 255] = img[mask == 255]
        if output_file:
            cv2.imwrite(output_file, out)
        return out

    @staticmethod
    def contour_approx(c):
        epsilon = 0.01*cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        return approx if len(approx) >= 4 else None # look for edges with more than 3 points

    def find_nested_contours(self, image):
        it = 1
        th2 = cv2.dilate(image, np.ones((3,3), np.uint8), iterations=it)
        cv2.imwrite('data/tmp/03_th2.png', th2)
        contours = cv2.findContours(th2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        approxs = list(filter(lambda x: x is not None, map(self.contour_approx, contours)))
        approxs = list(filter(self.filter_contours, approxs))
        return list(filter(self.filter_contours, approxs))
    
    def find_biggest_contour(self, img):
        contours = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        approxs = list(filter(lambda x: x is not None, map(self.contour_approx, contours)))
        approxs = list(filter(self.filter_contours, approxs))

        if len(approxs):
            return sorted(approxs, key=cv2.contourArea, reverse=True)[0]
        return None


if __name__ == '__main__':
    SudokuDetector(data[1])
