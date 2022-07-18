# -*- coding: utf-8 -*-
"""
ITC6125A1 - MACHINE LEARNING & APPLICATIONS - SPRING TERM 2022
Term Project - Traffic Sign Classifier
Students:
    Ioannis Fitsopoulos - s-if257217
    Trafalis Panagiotis - s-pt256311
Instructor: Milioris Dimitrios

"""

import cv2
import keras
import numpy as np
from traffic_sign_classifier import *
from class_names import classname
import time

MODEL_PATH = 'C:\\Users\\johnfitsos\\Desktop\\mlapps\\Model_.model_'



framewidth=640
frameheight = 400
brightness =100
threshold = 0.90
font = cv2.FONT_HERSHEY_SIMPLEX\
    
#Setup the video camera
cap = cv2.VideoCapture(0)
cap.set(3,framewidth)
cap.set(3,frameheight)
cap.set(10, brightness)

model= keras.models.load_model(MODEL_PATH)
m=predict_sign()
m.m1=model

while True:
    sucess, imgOriginal = cap.read()
    img=read_img(img=imgOriginal)
    # img=np.asarray(imgOriginal)
    
    input_img = prepare_input(img)
    
    results = m.predict_class(input_img)
    prob,class_ = output_class(results)
    print(f'With probability {round(prob,2)} give us class: {classname(class_)}')

    
    if prob > threshold:
        try:
            cv2.putText(imgOriginal, str(class_)+" "+str(classname(class_)), (20,35), font, 0.75, (255, 255, 0), 2,cv2.LINE_AA)
            text = f"""{round(prob*100,2)}% """
            cv2.putText(imgOriginal, text,(20, 75), font, 0.75, (255, 255, 0), 2, cv2.LINE_AA)
        except:
            pass
    cv2.imshow("Result", imgOriginal)
    #time.sleep(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break