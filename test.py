import cv2 as cv
import numpy as np

capture = cv.VideoCapture(0)

while True:
    ret, frame = capture.read()
    width = int(capture.get(3))
    height = int(capture.get(4))

    hsv = cv.cvtColor(frame , cv.COLOR_BGR2HSV)
    lower_limit = np.array([136, 87, 111], np.uint8)
    upper_limit = np.array([180, 255, 255], np.uint8)
    mask = cv.inRange(hsv, lower_limit, upper_limit)
# test out the dilation feature to enhance the specified kernel recognitionx
    
    # kernel = np.ones((3,3) ,'uint8')
    kernel1 = np.ones((7,7) ,'uint8')
    mask1 = cv.dilate(mask ,kernel1)
    # mask = cv.dilate(mask ,kernel)
    
    #larger kernel size retains the coloured image better (i guesss)

    # resulting_image = cv.bitwise_and(frame, frame, mask=mask)
    resulting_image1 = cv.bitwise_and(frame, frame, mask=mask1)


    #making contours

    contours , heirarchies = cv.findContours(mask1, cv.RETR_TREE , cv.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv.contourArea(contour)
        if(area > 4500):
            x, y, w, h = cv.boundingRect(contour)
            frame = cv.rectangle(frame, (x, y), 
                                       (x + w, y + h), 
                                       (0, 0, 255) , 2)
              
            cv.putText(frame, "RED", (x, y),
                        cv.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255)) 

    # cv.imshow('Frame' , mask)
    # cv.imshow('Frame' , resulting_image)
    
    cv.imshow('Frame2' , frame)
    if cv.waitKey(1) == ord('x'):
        capture.release()
        cv.destroyAllWindows()
        break



