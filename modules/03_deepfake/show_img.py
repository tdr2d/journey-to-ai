import cv2
import sys
import os
import pathlib

if len(sys.argv) < 2:
    print("Usage python show_img.py 12.12.12.12:path/to/file.jpg")

path = sys.argv[1]
p = pathlib.Path(path)
cmd = os.system(f'scp {path} .')
img = cv2.imread(p.name)
cv2.imshow(p.name, img)
cv2.waitKey(0)
cv2.destroyAllWindows()