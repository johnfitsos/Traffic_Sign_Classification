"""
ITC6125A1 - MACHINE LEARNING & APPLICATIONS - SPRING TERM 2022
Term Project - Traffic Sign Classifier
Students:
    Ioannis Fitsopoulos - s-if257217
    Trafalis Panagiotis - s-pt256311
Instructor: Milioris Dimitrios

"""


from tensorflow import keras
import cv2
import numpy as np
import pandas as pd
from class_names import classname

# Insert your paths for the saved model and the image you want to classify

MODEL_PATH = 'C:\\Users\\johnfitsos\\Desktop\\mlapps\\Model_.model_'
IMG_PATH = 'C:\\Users\\johnfitsos\\Downloads\\STOP_sign.jpg'


def read_model(_path_):
    """
    Loads model from saved file
    """
    model = keras.models.load_model(_path_)
    return model


def read_img(_path_=None,frompath=False, img=None):
    """
    Reads and preprocess image, inline with the accepted inputs of the model
    """
    if frompath == True:
        
        img = cv2.imread(_path_)
    
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    print(img.shape)
    img = cv2.resize(img,(100,100))
    img = img/255
    return img



def prepare_input(img):
    """
    Returns an array of the image, ready to be provided as input to keras model
    """
    return np.expand_dims(img, axis=0)


def output_class(results):
    """ Takes pandas dataframe as input and returns the class name of the most probable sign
"""
    prob,class_= results.values[0][0],results.values[0][1]
    return prob,class_ 
    
class predict_sign:
    
  def __init__ (self,m1=None):
    """ Initialize the model """
    self.m1=m1 # Model

  def predict_class(self,photo):
    """ 
    Input: Array of an iamge with shape (100,100)
    
    Return: A dataframe with the probability of each class of signs in descending order
    
    - The first class of the dataframe is the most probable sign
    
    """
    a=list(self.m1.predict(photo)[0])
    b=[i for i in range(43)]
    c = pd.DataFrame(list(map(lambda x,y: [x,y],a,b))).rename(columns={0:'prob', 1:'class'}).sort_values(by='prob', ascending=False)
    return c 

if  __name__=="__main__":
    m=predict_sign()
    model = read_model(MODEL_PATH)
    img = read_img(_path_=IMG_PATH, frompath=True)
    input_img = prepare_input(img)
    m.m1=model
    results = m.predict_class(input_img)
    prob,class_ = output_class(results)
    print(results)
    print(f'With probability {round(prob,2)} give us class: {classname(class_)}')



