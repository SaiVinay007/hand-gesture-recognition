import cv2
import numpy as np
from matplotlib import pyplot as plt

Hand_image = cv2.imread('/home/saivinay/Documents/hand_gesture/five-hand-image.jpg')
image_YCrCb_format = cv2.cvtColor(Hand_image,cv2.COLOR_BGR2YCrCb)

#  Taking the general range of skin color 
lower_skin_color = np.array([80,135,85])
upper_skin_color = np.array([255,180,135])

mask = cv2.inRange(image_YCrCb_format,lower_skin_color,upper_skin_color)
roi_image = cv2.bitwise_and(image_YCrCb_format,image_YCrCb_format,mask = mask)

edges = cv2.Canny(mask,100,200)

# To make the un conneted points connected to the nearest points
gaussian = cv2.GaussianBlur(mask,(5,5),0)

# Returns all the contours present
image, contours, hierarchy = cv2.findContours(gaussian,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


# Considering the maximum of the contour areas present 
max = 0

for i,contour in enumerate(contours):
    num = 0
    if max < len(contour):
        max  = len(contour)
        num = i


max_area = cv2.contourArea(contours[num])
print(max_area)


draw_contours = cv2.drawContours(roi_image, contours, num, (0,255,0), 3)
print(len(draw_contours))

cv2.imshow('gaussian',gaussian)
# cv2.imshow('image_YCrCb_format',image_YCrCb_format)
# cv2.imshow('mask',mask)
cv2.imshow('roi_image',roi_image)
# cv2.imshow('draw_contours',draw_contours)
# cv2.imshow('edges',edges)

k = cv2.waitKey(70000) & 0xFF
cv2.destroyAllWindows()