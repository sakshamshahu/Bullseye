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

    resulting_image = cv.bitwise_and(frame, frame, mask=mask)
    cv.imshow('Frame' , mask)
    cv.imshow('Frame' , resulting_image)

    

    if cv.waitKey(1) == ord('x'):
        break

capture.release()
cv.destroyAllWindows()
