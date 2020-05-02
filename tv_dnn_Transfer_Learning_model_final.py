#!/usr/bin/env python
# coding: utf-8

# ## Setting Random Seed

# In[97]:


from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


# ## GPU

# In[98]:


# ## GPU
import os
import tensorflow as tf
import keras
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#cpu-gpu configuration
#gpu_options = tf.GPUOptions(visible_device_list="5,6")
os.environ["CUDA_VISIBLE_DEVICES"]="4"

config = tf.ConfigProto(device_count = {'GPU':2, 'CPU':4}) #max no of GPUs = 1, CPUs =4
#config = tf.ConfigProto(gpu_options=gpu_options)

#config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
keras.backend.set_session(sess)


# ## Importing Libraries

# In[99]:


import numpy as np
import re
import pandas as pd
from numpy import array
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import ast 
import joblib
import math
import time
current_t = time.time()
from pandas import DataFrame
from array import array
import xgboost 
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import MinMaxScaler
#from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_squared_error as mse
from sklearn.feature_selection import VarianceThreshold
import math
import sklearn
from pandas import DataFrame
import pickle
import scipy
from scipy import sparse
import pyodbc
import seaborn as sb
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import preprocessing
import os
from sklearn.metrics import roc_auc_score  
from scipy.sparse import csr_matrix
from scipy.stats import randint as sp_randint
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, make_scorer
import warnings
warnings.filterwarnings('ignore')
#from termcolor import colored
from sklearn.metrics import classification_report
from multiprocessing import Pool
from timeit import default_timer as timer
from math import sqrt
from collections import defaultdict
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import SelectPercentile, f_classif
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from scipy.stats import uniform as sp_rand
from sklearn import metrics   #Additional scklearn functions
from sklearn.model_selection import cross_validate
from sklearn.metrics.scorer import make_scorer
from sklearn.metrics import confusion_matrix
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg") #Needed to save figures
from sklearn.metrics import roc_auc_score
import sklearn.metrics
import json
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
import torch
import time
import numpy as np
import pandas as pd
import cv2 as cv
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize, CascadeClassifier
import glob
from tkinter import *
from PIL import Image, ImageTk
import os
import time, sys
from tkinter import font
import time
import random


# ## Import keras models for Neural Network training

# In[100]:


from keras.preprocessing import image
from keras.models import Model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Input, Activation, add, Dense, Flatten, Dropout, Multiply, Embedding, Lambda
from keras.layers import Conv2D, MaxPooling2D,PReLU
from keras import backend as K
from keras.utils.vis_utils import plot_model
import theano
from keras.layers import Dense, Convolution2D, UpSampling2D, MaxPooling2D, ZeroPadding2D, Flatten, Dropout, Reshape
from keras.models import Sequential
from keras.utils import np_utils
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
from keras.utils import Sequence
from keras.utils.np_utils import to_categorical
from keras.layers import LSTM, Dense, Input, Masking, Flatten, Dropout, BatchNormalization
from keras.optimizers import RMSprop
from keras.models import load_model
from keras import regularizers


# ## Read training data files and append to a dataframe

# In[ ]:



# df = pd.DataFrame()
# path = r'/mnt/sde/jagadish/userdata/dl_project/tv_train_files_updated_f/' # use your path
# all_files = glob.glob(path + "/*.csv")

# li = []

# for filename in (all_files):
#     dd = pd.read_csv(filename, index_col=None, header=0)
#     dd['Video'] = filename.split('/')[7].split('_')[1].split('.')[0]
#     li.append(dd)

# df = pd.concat(li, axis=0, ignore_index=True)


# In[ ]:


# df.head()


# In[ ]:


# df.columns.values[0] = 'ind'


# In[ ]:


# df['index'] = df.index


# In[ ]:


# df.head(1)


# In[ ]:


# df= df.sort_values(by=['Video','index'])


# In[ ]:


#df.head()


# ## Extracting Features from Transfer Learning model and combining them with 2D landmark features

# ## Convolution Base

# In[101]:


model_tf = load_model('/mnt/sde/jagadish/userdata/dl_project/inceptionv3-ft.model')


# In[102]:


model_tf.summary()


# In[103]:


original_model    = model_tf
base_input  = original_model.get_layer(index=0).input
base_output = original_model.get_layer(index=-2).output
base_model  = Model(inputs=base_input, outputs=base_output)


# In[104]:


for layer in base_model.layers:
    layer.trainable = False


# In[105]:


print(model_tf.get_layer(index=0).input)


# In[106]:


print(model_tf.get_layer(index=-2).output)


# ## Get features and labels

# In[ ]:


def get_feature_label(data):
    # remove outliers
    #data_after = data[(data['price']<400) & (data['price']>1)]
    #data_after = data[data['price']>1]
    # split features and labels
    #train_features = data.drop(['responded'],axis=1)
    train_features = data.drop(['Y'],axis=1)
    train_labels = data.Y
    return train_features,train_labels


# ## Training Features

# In[ ]:


# df.head()


# In[ ]:


# df['key'] = df['Video'].astype(str) + df['ind'].astype(str)


# In[ ]:


# df.head(1)


# In[ ]:


# df = df.drop(columns='Video')
# df = df.drop(columns='index')
# df = df.drop(columns='ind')


# In[ ]:


# df.head(1)


# In[ ]:


# df['key'] = df['key'].astype(int)


# In[ ]:


# gf = pd.DataFrame()
# gf = pd.DataFrame(columns=['Video','Frame', 'Features'])


# In[ ]:


# import cv2
# import numpy
# import glob
# import pylab as plt

# folders = glob.glob('/mnt/sde/jagadish/userdata/dl_project/tv_train_videos_updated_final/*')
# imagenames_list = []
# count = 0
# for folder in folders:
#     for f in glob.glob(folder+'/*.jpg'):
#         img = cv.imread(f)
#         img = cv.resize(img, (299,299), cv.INTER_CUBIC)  
#         img = img.astype(np.float32)
#         img = img/255.0
#         img = np.expand_dims(img, axis=0)
#         img = img.reshape(1,299, 299,3)
#         features =  bottleneck_model.predict(img)
#         temp = re.findall(r'\d+', f) 
#         res = list(map(int, temp))
#         imagenames_list.append(features)
#         gf = gf.append({'Video': res[0],'Frame': res[2], 'Features': features}, ignore_index=True)
#         print(count)
#         count += 1
        

# # read_images = []        

# # for image in imagenames_list:
# #     read_images.append(cv2.imread(image, cv2.IMREAD_GRAYSCALE))


# In[ ]:


# gf['Frame'] = gf['Frame'].astype(int)
# gf['Video'] = gf['Video'].astype(int)


# In[ ]:


# gf['index'] = gf.index


# In[ ]:


# gf['index'] = gf.index
# gf=gf.sort_values(by=['Video','Frame'])


# In[ ]:


# len(gf)


# In[ ]:


# gf.head()


# In[ ]:


# gf['key'] = gf['Video'].astype(str) + gf['Frame'].astype(str)


# In[ ]:


# gf['key'] = gf['key'].astype(int)


# In[ ]:


# gf.head(1)


# In[ ]:


# gf = gf.drop(columns='Video')
# gf = gf.drop(columns='index')
# gf = gf.drop(columns='Frame')


# In[ ]:


# gf.head(1)


# In[ ]:


# mf = pd.merge(left = df,right = gf, left_on = 'key', right_on ='key')


# In[ ]:


# mf.head(1)


# In[ ]:


# mf = mf.drop(columns='key')


# In[ ]:


##df.drop(list(df.filter(regex = 'fd')), axis = 1, inplace = True)


# In[ ]:


# mf.head(1)


# In[ ]:


# tf = list(mf['Features'])


# In[ ]:


# np.shape(tf)


# In[ ]:


# s = np.shape(tf)


# In[ ]:


# tf = np.reshape(tf,(s[0],s[2]))


# In[ ]:


# np.shape(tf)


# In[ ]:


# mf = mf.drop(columns='Features')


# In[ ]:


# mf.head(1)


# In[ ]:


# train_features,train_labels=get_feature_label(mf)
# train_features=train_features
# train_labels=train_labels


# In[ ]:


# X = train_features
# y = train_labels


# In[ ]:


# X.shape


# In[ ]:


# type(X)


# In[ ]:


# np.shape(tf)


# In[ ]:


# lm = pd.DataFrame(tf)


# In[ ]:


# lm.shape


# ## Combining  2D landmark  training features and features from pretrained Inception_V3 network

# In[ ]:


# X_tr = pd.concat([X,lm],axis=1)


# In[ ]:


# X_tr.shape


# In[ ]:


# y.shape


# ## Saving Training Features and Labels

# In[ ]:


# from sklearn.externals import joblib
# filename = 'tv_train_comb_f_6.sav'
# joblib.dump(X_t, filename)


# In[ ]:


# from sklearn.externals import joblib
# filename = 'tv_train_comb_l_6.sav'
# joblib.dump(y_t, filename)


# ## Loading Training Features and Labels

# In[5]:


from sklearn.externals import joblib
filename = 'tv_train_comb_f_6.sav'
X_tr = joblib.load(filename)


# In[6]:


from sklearn.externals import joblib
filename = 'tv_train_comb_l_6.sav'
y = joblib.load(filename)


# In[ ]:





# ## Read test data

# In[33]:


path = sys.argv[1]


# In[35]:


path.split('/')[-1].split('.')[0]


# In[34]:


count = 0
cap = cv.VideoCapture(path)   # capturing the video from the given path
#cap.set(cv.CAP_PROP_FPS, 15)

video_sequence =[]
face_video =[]  
while(cap.isOpened()):
  frameId = cap.get(1) #current frame number
  ret, frame = cap.read()
  if (ret != True):
      break


  filename1 ="frame_%d.jpg" % count

  directory = path.split('/')[-1].split('.')[0]
  #directory = 'vid'


  parent_dir = './test_videos_frames'

  path1 = os.path.join(parent_dir, directory) 
  os.makedirs(path1, exist_ok=True)


  cv.imwrite(os.path.join(path1 , filename1), frame)



  count+=1



cap.release()


print (count)


# In[ ]:


python OpenPose.py path


# In[62]:


directory = path.split('/')[-1].split('.')[0]



parent_dir = "./test_videos_json_files"

path2 = os.path.join(parent_dir, directory) 


# In[ ]:


python parse_json_tv.py path2


# In[80]:


f = path.split('/')[-1].split('.')[0]



parent_dir = "./test_data"
#parent_dir = "/mnt/sde/jagadish/userdata/dl_project/tv_test_data/"

filename = os.path.join(parent_dir + f + '.csv')


# In[82]:


test = pd.read_table(filename,sep=",")


# In[83]:


len(test)


# In[84]:


pf = test


# In[ ]:


pf.drop(list(pf.filter(regex = 'fd')), axis = 1, inplace = True)


# In[ ]:


pf.head(1)


# In[ ]:


pf.columns.values[0] = 'ind'


# In[ ]:


pf['index'] = pf.index


# In[ ]:


pf.head(1)


# In[ ]:


pf['key'] = pf['index'].astype(str)


# In[ ]:


pf.head(1)


# In[ ]:


pf['key'] = pf['key'].astype(int)


# In[ ]:


pf = pf.drop(columns='ind')
pf = pf.drop(columns='index')


# In[ ]:


pf.head(1)


# ## Extracting features from transfer learning model for test data

# In[88]:


kf = pd.DataFrame()
kf = pd.DataFrame(columns=['Frame', 'Features'])


# In[90]:


directory = path.split('/')[-1].split('.')[0]
#directory = '9038'


parent_dir = './test_videos_frames'
#parent_dir = '/mnt/sde/jagadish/userdata/dl_project/tv_test_videos'

path_n = os.path.join(parent_dir, directory) 


# In[109]:


filename="/mnt/sde/jagadish/userdata/dl_project/tv_test_videos/9036/img_9036_frame_36.jpg"


# In[ ]:


path5 = path_n # use your path
all_files = glob.glob(path5 + "/*.jpg")

li = []
count=0
for filename in (all_files):
        img = cv.imread(filename)
        img = cv.resize(img, (299,299), cv.INTER_CUBIC)  
        img = img.astype(np.float32)
        img = img/255.0
        img = np.expand_dims(img, axis=0)
        img = img.reshape(1,299, 299,3)
        features =  base_model.predict(img)
        temp = re.findall(r'\d+', filename) 
        res = list(map(int, temp))
        a = filename.split('/')[8].split('_')[0]
        b = filename.split('_')[5].split('.')[0]
        kf = kf.append({'Frame': res[-1], 'Features': features}, ignore_index=True)
        print(count)
        count += 1


# In[ ]:


kf.head(2)


# In[ ]:


len(kf)


# In[ ]:


kf['Frame'] = kf['Frame'].astype(int)


# In[ ]:


kf=kf.sort_values(by=['Frame'])


# In[ ]:


kf.head(2)


# In[ ]:


kf.shape


# In[ ]:


kf['key'] = kf['Frame'].astype(str)


# In[ ]:


kf.head(1)


# In[ ]:


kf['key'] = kf['key'].astype(int)


# In[ ]:


nf = pd.merge(left = pf, right = kf, left_on = 'key', right_on = 'key')


# In[ ]:


nf.head(1)


# In[ ]:


nf = nf.drop(columns='Frame')


# In[ ]:


nf = nf.drop(columns='key')


# In[ ]:


nf.head(1)


# In[ ]:


bf = list(nf['Features'])


# In[ ]:


np.shape(bf)


# In[ ]:


s = np.shape(bf)


# In[ ]:


bf = np.reshape(bf,(s[0],s[2]))


# In[ ]:


np.shape(bf)


# In[ ]:


nf = nf.drop(columns='Features')


# In[ ]:


nf.head(1)


# In[ ]:


test_features = nf


# In[ ]:


test_features.shape


# In[ ]:


sm = pd.DataFrame(bf)


# In[ ]:


sm.shape


# In[ ]:


X_te = pd.concat([test_features,sm],axis=1)


# In[ ]:


X_te.shape


# In[ ]:


#test_labels.shape


# ## Saving Test Features and Labels

# In[ ]:


# from sklearn.externals import joblib
# filename = 'tv_test_8307_comb_f_3.sav'
# joblib.dump(X_te, filename)


# In[ ]:


# from sklearn.externals import joblib
# filename = 'tv_test_8340_comb_l_3.sav'
# joblib.dump(test_labels, filename)


# ## Loading Test features and Labels

# In[76]:


# from sklearn.externals import joblib
# filename = 'tv_test_8358_comb_f_3.sav'
# X_te = joblib.load(filename)


# In[77]:


# from sklearn.externals import joblib
# filename = 'tv_test_8358_comb_l_3.sav'
# test_labels = joblib.load(filename)


# In[ ]:


# X_te.shape


# In[ ]:


# test_labels.shape


# ## Final Normalizing

# In[7]:


from sklearn.preprocessing import MinMaxScaler


# In[8]:


scaler = MinMaxScaler()


# In[9]:


X_tr = scaler.fit_transform(X_tr)
X_tr = pd.DataFrame(X_tr)


# In[78]:


X_te = scaler.transform(X_te)
X_te = pd.DataFrame(X_te)


# In[ ]:





# ## Compile and fit the model

# In[10]:


# rmsprop = optimizers.RMSprop(lr=0.001)
# adam = optimizers.Adam(lr=0.0001)
# sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# ada =optimizers.Adadelta(lr=0.0001, rho = 0.95, epsilon = 1e-07)


# In[ ]:


# import time
# current_t = time.time()

# verbose, epochs, batch_size = 1, 11, 15
# n_features, n_outputs = 2768, 1
# # define model
# model = Sequential()
# #model.add(LSTM(500, activation='relu',return_sequences=False, input_shape=(n_samples, n_features)))
# model.add(Dense(200, activation='relu',
#                 kernel_regularizer=regularizers.l2(0.001), input_shape=(n_features,)))
# model.add(Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
# model.add(Dense(n_outputs, activation='sigmoid'))
# model.compile(loss='binary_crossentropy',metrics=['accuracy'], optimizer='Adam')

# model.summary()

# # fit network
# history = model.fit(X_tr, y, epochs=epochs, batch_size=batch_size,validation_split=0.00, verbose=verbose)



# ## Save model

# In[41]:




#model.save('tv_model_tf_20_v6.h5')  # creates a HDF5 file 'tv_model.h5'


# ## Load the saved model

# In[83]:


# returns a compiled model
# identical to the previous one
#model = load_model('tv_model_1200_u1.h5')
model = load_model('./pretrained_model/tv_model_tf_20_v6.h5')


# ## Plotting the results

# In[84]:


# import matplotlib.pyplot as plt
# #acc = history.history['acc']
# #val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs = range(1, len(loss) + 1)

# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.savefig('Watching_TV_train_val_loss_curve.jpg')  # saves the current figure
# plt.show()


# In[85]:


# import matplotlib.pyplot as plt
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs = range(1, len(acc) + 1)

# plt.plot(epochs, acc, 'go', label='Training accuracy')
# plt.plot(epochs, val_acc, 'g', label='Validation accuracy')
# plt.title('Training and validation accuracy')
# plt.legend()
# plt.savefig('Watching_TV_train_val_accuracy_curve.jpg')  # saves the current figure
# plt.show()


# ## Testing the model

# In[86]:


y_p = model.predict(X_te, verbose=0)
results = y_p


# ## Classification Metrics

# In[87]:


results[results<=0.5]=0
results[results>0.5]=1


# In[88]:


# # Creating the Confusion Matrix
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(test_labels, results)


# In[89]:


# cm


# In[74]:


# y_pred = results
# y_true = test_labels


# In[75]:


# accuracy = format(accuracy_score(y_true, y_pred),'.4f')


# sensitivity = format(recall_score(y_true, y_pred,pos_label=1,average='binary'),'.4f')

# specificity = format(recall_score(y_true, y_pred,pos_label=0,average='binary'),'.4f')

# print('Accuracy : ', accuracy)   
# print('Sensitivity : ', sensitivity)
# print('Specificity : ', specificity)


# In[ ]:


#print ("Features_extraction complete. Time elapsed: " + str(int(time.time()-current_t )) + "s")


# ## Save JSON file with time and label information

# In[ ]:


kf = results


# In[ ]:


hf = pd.DataFrame(kf)


# In[ ]:


mf = pd.DataFrame(columns=['Watching_TV'])


# In[ ]:


cap = cv.VideoCapture(path)   # capturing the video from the given path
fps = cap.get(cv.CAP_PROP_FPS) # Getting Franme rate of the video


# In[ ]:


fps


# In[ ]:


n= hf.index
l=[]
c=0
for i in n[:] :
    
    l.append(c/fps)
    l.append(hf.iloc[i][0])
    
    mf = mf.append({'Watching_TV':l[:]}, ignore_index=True)
    l=[]
    c+=1


# In[ ]:


mf.head()


# In[ ]:


mf.to_json('timeLable.json')


# ## Plot and save "Time vs Label" graph

# In[ ]:


pf = pd.DataFrame(columns=['Time', 'Label'])


# In[ ]:


n= hf.index
c=0
for i in n[:] :
    

    
    pf = pf.append({'Time': c/fps, 'Label': hf.iloc[i][0]}, ignore_index=True)

    c+=1


# In[ ]:


pf.head()


# In[ ]:


time = pf['Time']
label1 = pf['Label']


# In[ ]:


plt.figure(figsize=(20,10))
plt.plot(time, label1, 'g')
#plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.xticks(fontsize=20, fontweight='bold',rotation=90)
plt.yticks(fontsize=20, fontweight='bold')
plt.xlabel('Time (seconds)',fontsize=20, fontweight='bold')
plt.ylabel('Label',fontsize=20, fontweight='bold')
plt.title('Time vs Label', fontsize=20, fontweight='bold')
plt.tight_layout()
#plt.legend()
plt.savefig('timeLable.jpg')  # saves the current figure
plt.show()


# In[ ]:





# In[ ]:




