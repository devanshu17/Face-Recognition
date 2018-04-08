
## whenever the program captures a face we will write that in a folder
## Befroe capturing the face we need to tell the script whose face it is
## for that we will3 use an identifier called "id"
## use the below script to create the dataset creator

import cv2
import numpy as np

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml'); #detect is our image classifier
cam = cv2.VideoCapture(0) #cam is videocapture object

id = raw_input('Enter user ID') #identifier; store this ID with the corresponding face
sampleNum = 0;

while(True):
    ret, img = cam.read(); #cam.read returns status variable and capture image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #gray scale image to detect faces
    faces = faceDetect.detectMultiScale(gray,1.3,5); #list to detect faces; detect all faces in curret frame and return coordinates of the faces 

    for (x,y,w,h) in faces:
        sampleNum = sampleNum + 1;
        cv2.imwrite("dataSet/User." + str(id) + "." + str(sampleNum) + ".jpg",gray[y:y+h,x:x+w]) #write the captured face ina file
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2) #drawing image
        cv2.waitKey(100);
    cv2.imshow("Face",img); #to show window
    #if(cv2.waitKey(1)==ord('q')):
    #    break;
    cv2.waitKey(1);
    if(sampleNum>30):
        break;

cam.release()
cv2.destroyAllWindows()    
    
