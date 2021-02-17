# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 11:12:51 2021

@author: PINTU
"""

import cv2
import urllib
from tensorflow.keras.models import load_model
import numpy as np

model = load_model("Face_Recognition_Final_Project.h5")

classifier = cv2.CascadeClassifier(r"haarcascade_frontalface_default.xml")

url = "http://192.168.43.1:8080/shot.jpg"

def get_pred_label(pred):
    labels = ['Asutosh','Biswajit','Hritik','Soumya']
    return labels[pred]
    
def preprocess(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(100,100))
    img = img.reshape(1,100,100,1)
    return img


while True:
    image_from_url = urllib.request.urlopen(url)
    frame = np.array(bytearray(image_from_url.read()),np.uint8)
    frame = cv2.imdecode(frame,-1)
    
    faces = classifier.detectMultiScale(frame,1.5,5)
    
    for x,y,w,h in faces:
        face_frame = frame[y:y+h,x:x+w]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
        cv2.putText(frame,get_pred_label(model.predict_classes(preprocess(face_frame))[0]),(200,500),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)
        
    cv2.imshow("detector", frame)
    if cv2.waitKey(30)==ord("q"):
        break
        
cv2.destroyAllWindows()
        
        
        
        