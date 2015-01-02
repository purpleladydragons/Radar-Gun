import time
import cv2
from PIL import Image
import numpy as np
import subprocess as sp
# take video/picture frames

def findMovement(img, topaint):
    detected = False
    contours, hier = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x = 0
    y = 0
    if len(contours) > 0:
        detected = True

    hulls = [ x for x in contours ]
    
    if detected:
        for i in range(len(contours)):
            if len(contours[i]) > 50:
                hulls[i] = (cv2.convexHull(contours[i]))
                cv2.drawContours(topaint, hulls, i, 255, 3)
                #cv2.drawContours(img, contours, i, 255, 3)
                #rect = cv2.boundingRect(contours[i])
                #x = rect[0] + rect[2]/2
                #y = rect[1] + rect[3]/2
                #cv2.circle(topaint, (x,y), rect[3]/2, 255, 2)


kernel = np.ones((5,5), np.uint8)

cap = cv2.VideoCapture("kick.mp4")
count = 1
ret, avg = cap.read()
avg = cv2.cvtColor(avg, cv2.COLOR_BGR2GRAY)
avg = np.asarray(avg, dtype=np.int32)

while cap.isOpened():
    ret, frame = cap.read()
    if frame == None: break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg += frame
    count += 1

avg /= count
avg = np.asarray(avg, dtype=np.uint8)
avg = cv2.resize(avg, (1000, 600))

while True:
    cap = cv2.VideoCapture("kick.mp4")
    while cap.isOpened():
        ret, frame1 = cap.read()
        ret, frame2 = cap.read()
        if frame1 == None or frame2 == None:
            break
        frame1 = cv2.resize(frame1, (1000, 600))
        frame = frame1
        frame2 = cv2.resize(frame2, (1000, 600))
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        diff = cv2.absdiff(frame1, avg)
        ret, img = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        #img = cv2.dilate(img, kernel, iterations =5)
        #img = cv2.blur(img, (10,10))
        #img = cv2.erode(img, kernel, iterations =1)
        #ret, img = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
        findMovement(img, frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



#img = np.zeros_like(imgs[0])

#cv2.absdiff(imgs[5], imgs[6], img)
#ret, img = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
#img = cv2.blur(img, (5,5))
#ret, img = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)

#im = Image.fromarray(img)
#im.show()
# ask for ball type (soccer, football, basketball, baseball etc)

# detect ball in video (motion detection, edge detection, learn what a ball looks like?)
# we should be able to build up an aggregate image that shows how little the scene changes


# infer pixel-to-distance ratio from size of ball in pixels and known size of ball in real life

# using known framerate, r = d/t

ball_diameter = 12

