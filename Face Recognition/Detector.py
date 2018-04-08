import cv2
import numpy as np

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml'); #detect is our image classifier
cam = cv2.VideoCapture(0) #cam is videocapture object
rec = cv2.createLBPHFaceRecognizer(); #create a recognizer
rec.load("recognizer\\trainingData.yml") #load training data from the trained recognizer
id = 0
font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,5,1,0,4)
while(True):
    ret, img = cam.read(); #cam.read returns status variable and capture image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #gray scale image to detect faces
    faces = faceDetect.detectMultiScale(gray,1.3,5); #list to detect faces; detect all faces in curret frame and return coordinates of the faces 

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2) #drawing image
        id,config = rec.predict(gray[y:y+h,x:x+w]) #predict the id of faces
        if(id==1):
            id = 'Name_1'
        if(id==2):
            id = 'Name_2'
        else:
            id = 'Unknown'
    
        cv2.cv.PutText(cv2.cv.fromarray(img),str(id),(x,y+h),font,255); #print the above no.
    cv2.imshow("Face",img); #to show window
    if(cv2.waitKey(1)==ord('q')):
        break;

cam.release()
cv2.destroyAllWindows()    
    
