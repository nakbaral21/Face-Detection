import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

template1 = cv.imread('TemplateVideo.jpg')
template2 = cv.imread('TemplateVideo2.jpg')
template1 = cv.cvtColor(template1, cv.COLOR_BGR2GRAY)
template2 = cv.cvtColor(template2, cv.COLOR_BGR2GRAY)
w1, h1 = template1.shape[::-1]
w2, h2 = template2.shape[::-1]

vid = cv.VideoCapture(0)
while(True):
    ret,frame = vid.read()
    img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Apply template Matching
    res1 = cv.matchTemplate(img,template1,cv.TM_CCOEFF_NORMED)
    min_val1, max_val1, min_loc1, max_loc1 = cv.minMaxLoc(res1)
    res2 = cv.matchTemplate(img,template2,cv.TM_CCOEFF_NORMED)
    min_val2, max_val2, min_loc2, max_loc2 = cv.minMaxLoc(res2)

    top_left1 = max_loc1
    bottom_right1 = (top_left1[0] + w1, top_left1[1] + h1)
    top_left2 = max_loc2
    bottom_right2 = (top_left2[0] + w2, top_left2[1] + h2)

    cv.rectangle(img,top_left1, bottom_right1, 255, 2)
    cv.rectangle(img,top_left2, bottom_right2, 255, 2)

    # Hitung jarak
    mid_x1 = int((top_left1[0] + bottom_right1[0])/2)
    mid_x2 = int((top_left2[0] + bottom_right2[0])/2)
    dis_x = abs(mid_x1-mid_x2)
    mid_y1 = int((top_left1[1] + bottom_right1[1])/2)
    mid_y2 = int((top_left2[1] + bottom_right2[1])/2)
    dis_y = abs(mid_y1-mid_y2)
    dis_cm = int(-0.2564*dis_x+68.7179)
    
    cv.imshow('Hasil', img)
    print("Horizontal = ", dis_x, ", Vertikal = ", dis_y)
    print("Jarak = ", dis_cm, "cm")
    keyboard = cv.waitKey(38)
    if keyboard == 'q' or keyboard == 27:
        break

cv.destroyAllWindows()
