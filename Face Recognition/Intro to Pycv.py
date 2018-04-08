import cv2
import numpy as np

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml'); #detect is our image classifier
cam = cv2.VideoCapture(0) #cam is videocapture object

while(True):
    ret, img = cam.read(); #cam.read returns status variable and capture image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #gray scale image to detect faces
    faces = faceDetect.detectMultiScale(gray,1.3,5); #list to detect faces; detect all faces in curret frame and return coordinates of the faces 

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2) #drawing image
    cv2.imshow("Face",img); #to show window
    if(cv2.waitKey(1)==ord('q')):
        break;

cam.release()
cv2.destroyAllWindows()    
    
