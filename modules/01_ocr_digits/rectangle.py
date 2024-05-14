import cv2

class Rectangle:
    def __init__(self, x, y, w, h) -> None:
        self.x = x
        self.y = y
        self.w = w
        self.h = h
    
    def merge(self, rectangle):
        min_x = min(rectangle.x, self.x)
        min_y = min(rectangle.y, self.y)
        max_x = max(rectangle.x+rectangle.w, self.x+self.w)
        max_y = max(rectangle.y+rectangle.h, self.y+self.h)
        return Rectangle(min_x, min_y, max_x - min_x, max_y - min_y)

    def get_sub_image(self, img):
        return img[self.y:self.y + self.h, self.x:self.x + self.w]

    def center(self):
        return int(self.x + self.w / 2), int(self.y + self.h / 2)

    def debug(self, img, color=(0,255,0), stroke_size=1):
        cv2.rectangle(img,(self.x,self.y),(self.x+self.w,self.y+self.h),color,stroke_size)
