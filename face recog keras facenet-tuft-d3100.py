#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os
os.environ['PYTHONHASHSEED']=str(1)

import keras as K
import numpy as np
from keras.models import load_model
from mtcnn.mtcnn import MTCNN
from PIL import Image
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import os
import pandas as pd
from sklearn.svm import SVC
from keras_facenet import FaceNet
from keras.utils import  to_categorical
from sklearn.metrics import accuracy_score
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense,Dropout,Input


# In[2]:


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# In[3]:
def reset_random_seeds():
   os.environ['PYTHONHASHSEED']=str(1)
   tf.random.set_seed(1)
   np.random.seed(1)
   random.seed(1)

detect = MTCNN()
def preprocess_find_face(img_path):
    img = Image.open(img_path)
    img = img.convert('RGB')
    pixels = np.asarray(img)
    faces = detect.detect_faces(pixels)
    #print(img_path)
    if (faces==[]):
        return []
    x1,y1,w,h=faces[0]['box']
    x1 = abs(x1)
    y1 = abs(y1)
    x2 = abs(x1 + w)
    y2 = abs(y1 + h)
    face_box = pixels[y1:y2, x1:x2]
    img_face = Image.fromarray(face_box, 'RGB')
    img_face = img_face.resize((160, 160))
    face_arr = np.asarray(img_face)
    return face_arr


# In[4]:


facenet=FaceNet()
#print(facenet.summary())


# In[5]:


def face_embedding(model, face):
    face = face.astype('float32')
    mean=face.mean()
    std=face.std()
    face=(face-mean)/std
    fn_inp = np.expand_dims(face, axis=0)
    y = model.embeddings(fn_inp)
    return y[0]


# In[6]:


x,y=[],[]
dir_path="tuft"


# In[7]:


def make_dataset(d_path):
    subdir=os.listdir(d_path)
    for sub in tqdm(subdir):
        sub_path=os.path.join(d_path,sub)
        images=os.listdir(sub_path)
        for img in images:
            img_path=os.path.join(sub_path,img)
            img_processed=preprocess_find_face(img_path)
            if(img_processed!=[]):
                x.append(img_processed)
                y.append(int(sub))
    return np.asarray(x),np.asarray(y)


# In[8]:


'''X_train,y_train=make_dataset(dir_path)
np.savez_compressed("tuft.npz",X_train,y_train)'''


# In[9]:


data=np.load("tuft.npz")
X_train,y_train=data["arr_0"],data["arr_1"]


# In[10]:


X_train.shape


# In[11]:


'''X_train_embed=[]
for img in tqdm(X_train):
    img_embed=face_embedding(facenet,img)
    X_train_embed.append(img_embed)
X_train_embed=np.asarray(X_train_embed)


# In[12]:


np.savez_compressed("tuft_embed.npz",X_train_embed,y_train)'''


# In[13]:


data=np.load("tuft_embed.npz")
X_train_embed,y_train=data["arr_0"],data["arr_1"]
print(X_train_embed.shape)

# In[14]:
y_train=y_train.reshape((y_train.shape[0],1))
y_train=to_categorical(y_train)
print(y_train.shape)

X_train_embed=X_train_embed.astype('float32')
y_train=y_train.astype('float32')

print(X_train_embed[1][:])

X_train_e, X_test_e, y_train_e, y_test = train_test_split(X_train_embed, y_train, test_size=0.10)


# In[15]:


norm=Normalizer(norm="l2")


# In[16]:


X_train_e=norm.fit_transform(X_train_e)
X_test_e=norm.transform(X_test_e)


# In[17]:


#print(X_train_e)


# In[18]:

print(y_train_e.shape)


def make_model(out_layer):
    model=Sequential()
    model.add(Dense(units=512,activation='relu',input_shape=(512,)))
    model.add(Dropout(0.2))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=128, activation='relu'))
    '''if(out_layer<=64):
        model.add(Dense(unit=64,activation='relu'))'''
    model.add(Dense(units=out_layer,activation='softmax'))

    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')

    return model




pred_model=make_model(y_train.shape[1])
print(pred_model.summary())
pred_model.fit(X_train_e,y_train_e,validation_data=(X_test_e,y_test),epochs=50)

# In[19]:


y_train_pred=pred_model.predict(X_train_e)
print("accuracy ",accuracy_score(y_train_e,y_train_pred))


# In[20]:


y_test_pred=pred_model.predict(X_test_e)
print("accuracy ",accuracy_score(y_test,y_test_pred))


# video cam

# In[21]:


'''x=[]
y=[]
def make_dataset2(d_path):
    subdir=os.listdir(d_path)
    for sub in subdir:
        sub_path=os.path.join(d_path,sub)
        images=os.listdir(sub_path)
        for img in images:
            img_path=os.path.join(sub_path,img)
            img_processed=preprocess_find_face(img_path)
            if(img_processed!=[]):
                x.append(img_processed)
                y.append(sub)
    return np.asarray(x),np.asarray(y)


# In[23]:


dir_path2="myfaces"
X_train2,y_train2=make_dataset2(dir_path2)
np.savez_compressed("my_faces.npz",X_train2,y_train2)
#data=np.load("my_faces.npz")
#X_train2,y_train2=data["arr_0"],data["arr_1"]


# In[24]:


X_train2.shape


# In[25]:


X_train_embed2=[]
for img in X_train2:
    img_embed=face_embedding(facenet,img)
    X_train_embed2.append(img_embed)
X_train_embed2=np.asarray(X_train_embed2)


# In[26]:


#print(X_train_embed2.shape)


# In[27]:


np.savez_compressed("my_faces_embed.npz",X_train_embed2,y_train2)
#data=np.load("my_faces_embed.npz")
#X_train_embed2,y_train2=data["arr_0"],data["arr_1"]

# In[28]:
norm2=Normalizer()
norm2.fit_transform(X_train_embed2)


labeler=LabelEncoder()
y_train2=labeler.fit_transform(y_train2)


# In[29]:


pred_model2=SVC(kernel='linear')
pred_model2.fit(X_train_embed2,y_train2)



# In[31]:


y_train_pred2=pred_model2.predict(X_train_embed2)
print("accuracy ",accuracy_score(y_train2,y_train_pred2))


# In[32]:


cap=cv2.VideoCapture(0)


# In[33]:


while(True):
    ret,img=cap.read()
    img=Image.fromarray(img,'RGB')
    img = img.convert('RGB')
    pixels = np.asarray(img)
    img = np.asarray(img)
    faces = detect.detect_faces(pixels)
    if(faces==[]):
        continue
    print(faces[0]['box'])
    x,y,w,h=faces[0]['box']
    print(x)
    print(y)
    print(w)
    print(h)
    x1=x
    y1=y

    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
    face_box=pixels[y:y+h,x:x+w]
    img_face = Image.fromarray(face_box, 'RGB')
    img_face = img_face.resize((160, 160))
    face_arr = np.asarray(img_face)
    #face_arr=np.expand_dims(face_arr,axis=0)
    face_embed=face_embedding(facenet,face_arr)
    #X=norm2.transform(face_embed.reshape(-1, 128))
    X=face_embed.reshape(-1, 128)
    y=pred_model2.predict(X)
    #chance=max(pred_model2.predict_proba(X))
    y=labeler.inverse_transform(y)
    print(y)
    string=y[0]
    print (string)
    #print(chance)
    cv2.putText(img, string, (x1+w+10,y1+h+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

    cv2.imshow('img', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
        break
cap.release()
cv2.destroyAllWindows()


# In[ ]:'''




