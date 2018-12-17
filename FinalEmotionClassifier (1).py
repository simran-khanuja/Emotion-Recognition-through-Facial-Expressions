#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2


# In[2]:


#initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# In[3]:


#getting the corresponding file names
filenames = []
import glob
for file in glob.glob("AllEmotionLabelsCK+/*.txt"):
    filenames.append(file[20:37])    


# In[4]:


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
        
        """(x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # show the face number
        cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

    # show the output image with the face detections + facial landmarks
    cv2.imshow("Output", image)
    cv2.waitKey(0)"""


# In[5]:


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

        # show the face number
        """cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

    # show the output image with the face detections + facial landmarks
    cv2.imshow("Output", image)
    cv2.waitKey(0)"""


# In[ ]:


#all possible triangles in the final expression
triangles_all_327_final = {}
from itertools import combinations
for x in shapes_all_327_final.keys():
    triangles = list(combinations(shapes_all_327_final.get(x),3))
    triangles_all_327_final.update({x:triangles})


# In[ ]:


#all possible triangles in the neutral expression
triangles_all_327_neutral = {}
for x in shapes_all_327_neutral.keys():
    triangles = list(combinations(shapes_all_327_neutral.get(x),3))
    triangles_all_327_neutral.update({x:triangles})


# In[ ]:


# four features corresponding to each triangle stored in the format (a,b,alpha,beta)
# for final expression
features = []
features_all_327_final={}
import math
for x in list(triangles_all_327_final.keys())[:100]:
    for y in triangles_all_327_final.get(x):
        a = np.linalg.norm(y[0]-y[1])
        b = np.linalg.norm(y[0]-y[2])
        cosine_angle = np.dot((y[0]-y[1]),(y[2]-y[0])) / (a*b);
        angle = np.arccos(cosine_angle)
        alpha = np.degrees(angle)
        beta  = np.rad2deg(np.arctan2(y[2][1] - y[1][1], y[2][0] - y[1][0]))
        features.append([a,b,alpha,beta])
    features_all_327_final.update({x:features})
    features=[]


# In[ ]:


# four features corresponding to each triangle stored in the format (a,b,alpha,beta)
# for corresponding neutral expression
features = []
features_all_327_neutral = {}
import math
for x in list(triangles_all_327_neutral.keys())[:100]:
    for y in triangles_all_327_neutral.get(x):
        a = np.linalg.norm(y[0]-y[1])
        b = np.linalg.norm(y[0]-y[2])
        cosine_angle = np.dot((y[0]-y[1]),(y[2]-y[0])) / (a*b);
        angle = np.arccos(cosine_angle)
        alpha = np.degrees(angle)
        beta  = np.rad2deg(np.arctan2(y[2][1] - y[1][1], y[2][0] - y[1][0]))
        features.append([a,b,alpha,beta])
    features_all_327_neutral.update({x:features})
    features=[]


# In[ ]:


delta = []
delta_all = []
features_all_327 = {}
X=[]
for x in features_all_327_neutral.keys():
    for (i,y) in enumerate(features_all_327_neutral.get(x)):
        featureNeutral = y
        featureExpression = (features_all_327_final.get(x))[i]
        delta = [a - b for a, b in zip(featureExpression,featureNeutral)]
        delta_all.append(delta)
        delta=[]
    features_all_327.update({x:delta_all})
    X.append(delta_all)
    delta_all = []    


# In[ ]:


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


# In[ ]:


# scaling a,b,alpha,beta and assigning one value to each triangle
# CONFIRM THIS
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


# In[ ]:


X_final = pd.DataFrame(combinedFeatures)
Y_final = Y[:100]


# In[ ]:


X_final = X_final.apply(lambda x: x.fillna(x.mean()),axis=0)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_final,Y_final,test_size = 0.2,random_state=42)


# In[ ]:


# Feature Selection using Adaboost
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=200)
clf.fit(X_final,Y_final)
fi = list(clf.feature_importances_)


# In[ ]:


from sklearn.feature_selection import SelectFromModel
model = SelectFromModel(clf, prefit=True)
train_reduced = model.transform(X_train)


# In[ ]:


features = pd.DataFrame()
features['feature'] = X_train.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=False, inplace=True)


# In[ ]:


val = []
for i in range(0,len(train_reduced[1])):
    val.append(features['feature'].iloc[i])
    
reduced_test=[]
red_test=[]
for i in range(0,len(X_test)):
    for a in val:
        red_test.append(X_test[int(a)].iloc[i])
    reduced_test.append(red_test)
    red_test = []


# In[ ]:


# Choosing how many feautures to use for SVM classifier using gridsearch
"""from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score


parameters = {'n_estimators':[50,100,150,200]}    #Dictionary of parameters

scorer = make_scorer(accuracy_score)         #Initialize the scorer using make_scorer

grid_obj = GridSearchCV(clf,parameters,scoring=scorer)
#Initialize a GridSearchCV object with above parameters,scorer and classifier

grid_fit = grid_obj.fit(X_train,y_train)        
#Fit the gridsearch object with X_train,y_train

best_clf = grid_fit.best_estimator_        
#Get the best estimator. For this, check documentation of GridSearchCV object

numberOfFeatures = best_clf.get_params().get('n_estimators')
svmFeatures = fi[:numberOfFeatures]"""


# In[ ]:


# Training SVM classifier with reduced features

from sklearn import svm
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score

model=svm.SVC(kernel='rbf',C=1000.0).fit(train_reduced,y_train)
predictions=model.predict(reduced_test)

acc = accuracy_score(y_test, predictions)*100 


# In[ ]:


predictions


# In[ ]:


y_test


# In[ ]:


#0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise

