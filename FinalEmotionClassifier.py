#!/usr/bin/env python
# coding: utf-8

#0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise



# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import math



#initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")



#getting the corresponding file names
filenames = []
import glob
for file in glob.glob("AllEmotionLabelsCK+/*.txt"):
    filenames.append(file[20:37])    



#extracting 68 points for the emotional expression and storing it in a dictionary
shapes_all_327_final = {}
for x in filenames:
    image = cv2.imread("AllImagesCK+/%s.png" %(x))
    image = imutils.resize(image, width=500)
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # detect faces in the grayscale image
    rects = detector(gray, 1)
    
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        shapes_all_327_final.update({x:shape})
        


#extracting 68 points for the corresponding neutral expression and storing in dict
shapes_all_327_neutral = {}
for x in filenames:
    image = cv2.imread("AllImagesCK+/%s01.png" %(x[:-2]))
    image = imutils.resize(image, width=500)
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # detect faces in the grayscale image
    rects = detector(gray, 1)
    
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        shapes_all_327_neutral.update({x:shape})
        
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

       
#all possible triangles in the final expression
triangles_all_327_final = {}
from itertools import combinations
for x in shapes_all_327_final.keys():
    triangles = list(combinations(shapes_all_327_final.get(x),3))
    triangles_all_327_final.update({x:triangles})




#all possible triangles in the neutral expression
triangles_all_327_neutral = {}
for x in shapes_all_327_neutral.keys():
    triangles = list(combinations(shapes_all_327_neutral.get(x),3))
    triangles_all_327_neutral.update({x:triangles})


    
# four features corresponding to each triangle pair stored in the format (a(neutral)-a(expression),bn-be,alpha(n)-alpha(e)
# ,beta(n)-beta(e))
delta = []
delta_all = []
features_all_327 = {}
X=[]

for x in list(triangles_all_327_final.keys()):
    for (y1,y2) in zip(triangles_all_327_final.get(x),triangles_all_327_neutral.get(x)):
        a1 = np.linalg.norm(y1[0]-y1[1])
        b1 = np.linalg.norm(y1[0]-y1[2])
        cosine_angle1 = np.dot((y1[0]-y1[1]),(y1[2]-y1[0])) / (a1*b1);
        angle1 = np.arccos(cosine_angle1)
        alpha1 = np.degrees(angle1)
        beta1  = np.rad2deg(np.arctan2(y1[2][1] - y1[1][1], y1[2][0] - y1[1][0]))
        a2 = np.linalg.norm(y2[0]-y2[1])
        b2 = np.linalg.norm(y2[0]-y2[2])
        cosine_angle2 = np.dot((y2[0]-y2[1]),(y2[2]-y2[0])) / (a2*b2);
        angle2 = np.arccos(cosine_angle2)
        alpha2 = np.degrees(angle2)
        beta2  = np.rad2deg(np.arctan2(y2[2][1] - y2[1][1], y2[2][0] - y2[1][0])) 
        delta.extend((abs(a1-a2),abs(b1-b2),abs(alpha1-alpha2),abs(beta1-beta2)));
        delta_all.append(delta)
        delta=[]
    features_all_327.update({x:delta_all})
    X.append(delta_all)
    delta_all = []    





# making a dictionary of the emotion labels
Y = []
emotionLabels={}
for file in glob.glob("AllEmotionLabelsCK+/*.txt"):
    with open(file) as f:
        while True:
            c = f.read(1)
            if c!=' ':
                emotionLabels.update({file[20:37]:c})
                Y.append(c)
                break
    f.close()




# scaling a,b,alpha,beta and assigning one value to each triangle
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
combinedFeatures = []
com = []
scaler = MinMaxScaler()

for i in X:
    i = pd.DataFrame(i)
    i = scaler.fit_transform(i)
    for j in i:
        av = sum(j)/(float)(len(j))
        com.append(av)
    combinedFeatures.append(com)
    com=[]    


#Converting to Pandas DF and filling missing values
X_final = pd.DataFrame(combinedFeatures)
Y_final = Y
X_final = X_final.apply(lambda x: x.fillna(x.mean()),axis=0)



#Dropping columns with a variance below 5%
var_feat = []
for col in X_final.columns:
  if X_final[col].std()<0.05:
    var_feat.append(col)
X_final.drop(var_feat,axis=1,inplace=True)
    
    
    
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_final,Y_final,test_size = 0.2,random_state=42)





# Classification using SVM
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn import svm


model=svm.SVC(kernel='rbf',C=10.0,gamma=0.001,probability=True).fit(X_train,y_train)
predictions=model.predict(X_test)

print(accuracy_score(y_test, predictions)*100)

#0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise

