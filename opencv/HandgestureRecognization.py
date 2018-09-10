import cv2
import numpy as np
from matplotlib import pyplot as plt

######## From a normal image

# Hand_image = cv2.imread('/home/saivinay/Documents/hand_gesture/opencv/five.jpg')
# image_YCrCb_format = cv2.cvtColor(Hand_image,cv2.COLOR_BGR2YCrCb)

# #  Taking the general range of skin color 
# lower_skin_color = np.array([80,135,85])
# upper_skin_color = np.array([255,180,135])

# mask = cv2.inRange(image_YCrCb_format,lower_skin_color,upper_skin_color)
# roi_image = cv2.bitwise_and(image_YCrCb_format,image_YCrCb_format,mask = mask)

# edges = cv2.Canny(mask,100,200)

# # To make the un conneted points connected to the nearest points
# gaussian = cv2.GaussianBlur(mask,(5,5),0)


# # Returns all the contours present
# # Each individual contour is a Numpy array of (x,y) coordinates of boundary points of the object.
# # If you pass cv2.CHAIN_APPROX_NONE, all the boundary points are stored. But actually do we need all the points? 
# # For eg, you found the contour of a straight line. Do you need all the points on the line to represent that line? 
# # No, we need just two end points of that line. This is what cv2.CHAIN_APPROX_SIMPLE does. It removes all redundant points and compresses the contour, thereby saving memory           

# image, contours, hierarchy = cv2.findContours(gaussian,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


# # Considering the maximum of the contour areas present 
# max = 0

# for i,contour in enumerate(contours):
#     num = 0
#     if max < len(contour):
#         max  = len(contour)
#         num = i


# max_area = cv2.contourArea(contours[num])
# # print(max_area)


# # To draw all contours pass -1 in place of num
# draw_contours = roi_image
# cv2.drawContours(draw_contours, contours, num, (255,0,0), 3)
# # print(len(draw_contours))

# # Used for filling the space inside of the contour
# cv2.fillPoly(draw_contours, pts =[contours[num]], color=(255,255,255))

# check_convexity = gaussian

# # Here, cv.convexHull() function checks a curve for convexity defects and corrects it.
# # Generally speaking, convex curves are the curves which are always bulged out, or at-least flat.
# # And if it is bulged inside, it is called convexity defects.
# hull = cv2.convexHull(contours[num],returnPoints = False)


# # It returns an array where each row contains these values - [ start point, end point, farthest point, approximate distance to farthest point ].
# # We can visualize it using an image. We draw a line joining start point and end point, then draw a circle at the farthest point.
# # Remember first three values returned are indices of cnt. So we have to bring those values from cnt.
# defects = cv2.convexityDefects(contours[num],hull)

# print(defects)
# # print(hull)
# print(defects.shape[0])

# Number_of_fingers = 0
# for i in range(defects.shape[0]):
#     s,e,f,d = defects[i,0]
#     start = tuple(contours[num][s][0])
#     end = tuple(contours[num][e][0])
#     far = tuple(contours[num][f][0])
#     cv2.line(draw_contours,start,end,[0,255,0],2)
#     # print(start)
#     # cv2.circle(draw_contours,start,5,[0,0,255],-1)

#     if d > 10000 :
#         cv2.circle(draw_contours,far,5,[0,0,255],-1)
#         Number_of_fingers += 1

# print ("Number of fingers = "+str(Number_of_fingers) ) 

# cv2.imshow("origial_image",Hand_image)
# # cv2.imshow('gaussian',gaussian)
# # cv2.imshow('image_YCrCb_format',image_YCrCb_format)
# # cv2.imshow('mask',mask)
# # cv2.imshow('roi_image',roi_image)
# cv2.imshow('draw_contours',draw_contours)
# # cv2.imshow('edges',edges)

# k = cv2.waitKey(70000) & 0xFF
# cv2.destroyAllWindows()





##################################################################################################

## Taking images from video


cap = cv2.VideoCapture(0)

while(cap.read()):
    # for i in range(10):
    
    #     ret,frame = cap.read()
    #     frame += frame
    
    # frame = frame/10

    ret,frame = cap.read()
    # print(ret)

    image_YCrCb_format = cv2.cvtColor(frame,cv2.COLOR_BGR2YCrCb)

    #  Taking the general range of skin color 
    lower_skin_color = np.array([80,135,85])
    upper_skin_color = np.array([255,180,135])
    
    mask = cv2.inRange(image_YCrCb_format,lower_skin_color,upper_skin_color)
    roi_image = cv2.bitwise_and(image_YCrCb_format,image_YCrCb_format,mask = mask)
    
    edges = cv2.Canny(roi_image,100,200)
    
    # To make the un conneted points connected to the nearest points
    gaussian = cv2.GaussianBlur(mask,(5,5),0)
    
    
    # Returns all the contours present
    # Each individual contour is a Numpy array of (x,y) coordinates of boundary points of the object.
    # If you pass cv2.CHAIN_APPROX_NONE, all the boundary points are stored. But actually do we need all the points? 
    # For eg, you found the contour of a straight line. Do you need all the points on the line to represent that line? 
    # No, we need just two end points of that line. This is what cv2.CHAIN_APPROX_SIMPLE does. It removes all redundant points and compresses the contour, thereby saving memory           
    
    image, contours, hierarchy = cv2.findContours(gaussian,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    
    # Considering the maximum of the contour areas present 
    max = 0
    num = 0
    for i,contour in enumerate(contours):
        num = 0
        if max < len(contour):
            max  = len(contour)
            num = i
    
    # print(contours[num])
    max_area = cv2.contourArea(contours[num])
    # print(max_area)
    
    
    # To draw all contours pass -1 in place of num. Now it is drawing only for the largest contour
    draw_contours = roi_image
    cv2.drawContours(draw_contours, contours, num, (255,0,0), 3)
    # print(len(draw_contours))
    
    # Used for filling the space inside of the contour eith the given color
    cv2.fillPoly(draw_contours, pts =[contours[num]], color=(255,255,255))
    
    check_convexity = gaussian
    
    # Here, cv.convexHull() function checks a curve for convexity defects and corrects it.
    # Generally speaking, convex curves are the curves which are always bulged out, or at-least flat.
    # And if it is bulged inside, it is called convexity defects.
    hull = cv2.convexHull(contours[num],returnPoints = False)
    
    
    # It returns an array where each row contains these values - [ start point, end point, farthest point, approximate distance to farthest point ].
    # We can visualize it using an image. We draw a line joining start point and end point, then draw a circle at the farthest point.
    # Remember first three values returned are indices of cnt. So we have to bring those values from cnt.
    defects = cv2.convexityDefects(contours[num],hull)
    
    # print(defects.shape[0])
    # print(hull)
    # print(defects.shape[0])   
    
    Number_of_fingers = 0
    if defects is not None:
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(contours[num][s][0])
            end = tuple(contours[num][e][0])
            far = tuple(contours[num][f][0])
            cv2.line(draw_contours,start,end,[0,255,0],2)
            # print(start)
            # cv2.circle(draw_contours,start,5,[0,0,255],-1)

            if d > 10000 :
                cv2.circle(draw_contours,far,5,[0,0,255],-1)
                Number_of_fingers += 1
    
    if Number_of_fingers > 0:
        print ("Number of fingers = "+str(Number_of_fingers) ) 
    
    
    cv2.imshow("origial_image",frame)
    # cv2.imshow('gaussian',gaussian)
    cv2.imshow('image_YCrCb_format',image_YCrCb_format)
    cv2.imshow('mask',mask)
    # cv2.imshow('roi_image',roi_image)
    cv2.imshow('draw_contours',draw_contours)
    cv2.imshow('edges',edges)
    
    # k = cv2.waitKey(1) & 0xFF

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    
    
cap.release()  
cv2.destroyAllWindows()
 
    