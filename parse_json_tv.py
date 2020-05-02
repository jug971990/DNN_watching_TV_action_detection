#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[1]:


import pandas as pd
import time
import numpy as np
import math
from itertools import islice 
import json
from pandas.io.json import json_normalize
from sys import argv
import sys
import os


# ## File Path

# In[2]:


path = sys.argv[1]




# ## Read and parse json files for each video

# In[132]:




# this finds our json files
path_to_json = path
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]

# here I define my pandas Dataframe with the columns I want to get from the json
jsons_data = pd.DataFrame(columns=['index','person_id','po', 'hl', 'hr'])
df = pd.DataFrame()


# we need both the json and an index number so use enumerate()
for index, js in enumerate(json_files):
    with open(os.path.join(path_to_json, js)) as json_file:
        json_text = json.load(json_file)
        if json_text['people']!= []:
            index = (str(json_file).split('/')[-1].split('_')[2])
            person_id = json_text['people'][0]['person_id']
            po = json_text['people'][0]['pose_keypoints_2d']
            hl = json_text['people'][0]['hand_left_keypoints_2d']
            hr = json_text['people'][0]['hand_right_keypoints_2d'] 
            # here I push a list of data into a pandas DataFrame at row given by 'index'
            jsons_data.loc[index] = [index,person_id, po, hl, hr]


# now that we have the pertinent json data in our DataFrame let's look at it
#print(jsons_data)


# ## Preprocessing data

# In[133]:


df = jsons_data


# In[134]:


df['index'] = df['index'].astype(int)


# In[135]:


df=df.sort_index()


# In[136]:


df.head(5)


# In[137]:


len(df)


# In[138]:


gf = pd.DataFrame()


# In[139]:


gf=df


# In[140]:


gf.head(1)


# In[141]:


gf = gf.join(gf['po'].apply(pd.Series).add_prefix('po'))
gf = gf.join(gf['hl'].apply(pd.Series).add_prefix('hl'))
gf = gf.join(gf['hr'].apply(pd.Series).add_prefix('hr'))


# In[142]:


gf = gf.drop(['po', 'hl', 'hr', 'person_id', 'index'], axis=1)


# In[143]:


gf.head(1)


# In[144]:


gf = gf.drop(gf.iloc[:, 2::3],axis=1)


# In[145]:


gf.head(1)


# In[146]:


mf = pd.DataFrame()


# In[147]:


n = gf.shape[1]
i=0
j=0
while i < n:

    col = gf.columns[i][0:2] + "_" + str(j) #col name
    X = gf.columns[i]
    Y = gf.columns[i+1]
    mf[col] = gf[[X, Y]].values.tolist()
    i = i+2
    j= j+1


# In[148]:


mf.head(1)


# In[149]:


hf=pd.DataFrame()


# In[150]:


hf = mf


# In[151]:


hf.head(1)


# In[152]:


hf.shape[1]


# ## Euclidean distances between body key points

# In[153]:


def eudis5(v1, v2): # Function to calculate euclidean distance between two points
    dist = [(a - b)**2 for a, b in zip(v1, v2)]
    dist = math.sqrt(sum(dist))
    return dist


# In[154]:


hf['pd'] = ''
hf['pd'] = hf['pd'].apply(list)


# In[155]:


n=hf.index
m = hf.shape[1]
for i in n[:] :
    
    ear =[]
    I=1
    for j in range(25) :
        
        for k in range(I,25) :

            X = hf.columns[j]
            Y = hf.columns[k]

            a = np.array(hf[X][i])
            b = np.array(hf[Y][i])
            x = eudis5(a, b)
            ear.append(x)
        I = I + 1
     
    hf.loc[i,'pd'].append(ear[:])


# ## Euclidean distances between left hand key points

# In[156]:


hf['hld'] = ''
hf['hld'] = hf['hld'].apply(list)


# In[157]:


n=hf.index
m = hf.shape[1]
for i in n[:] :
    
    ear =[]
    I=26
    for j in range(25,46) :
        
        for k in range(I,46) :

            X = hf.columns[j]
            Y = hf.columns[k]

            a = np.array(hf[X][i])
            b = np.array(hf[Y][i])
            x = eudis5(a, b)
            ear.append(x)
        I = I + 1
     
    hf.loc[i,'hld'].append(ear[:])


# ## Euclidean distances between right hand key points

# In[158]:


hf['hrd'] = ''
hf['hrd'] = hf['hrd'].apply(list)


# In[159]:


n=hf.index
m = hf.shape[1]
for i in n[:] :
    
    ear =[]
    I=47
    for j in range(46,67) :
        
        for k in range(I,67) :

            X = hf.columns[j]
            Y = hf.columns[k]

            a = np.array(hf[X][i])
            b = np.array(hf[Y][i])
            x = eudis5(a, b)
            ear.append(x)
        I = I + 1
     
    hf.loc[i,'hrd'].append(ear[:])


# In[160]:


hf.head(5)


# ## Filtering the dataframe with desired columns

# In[161]:


df= pd.DataFrame()


# In[162]:


df=hf.filter(items=['pd',  'hld', 'hrd'])


# In[163]:


df.head(1)


# In[164]:


df = df.join(df['pd'].apply(pd.Series).add_prefix('p_'))
df = df.join(df['hld'].apply(pd.Series).add_prefix('hl_'))
df = df.join(df['hrd'].apply(pd.Series).add_prefix('hr_'))


# In[165]:


df.head(1)


# In[166]:


df = df.join(df['p_0'].apply(pd.Series).add_prefix('pd_'))
df = df.join(df['hl_0'].apply(pd.Series).add_prefix('hld_'))
df = df.join(df['hr_0'].apply(pd.Series).add_prefix('hrd_'))


# In[167]:


df.head(1)


# In[168]:


df = df.drop(['pd', 'hld', 'hrd','p_0', 'hl_0', 'hr_0'], axis=1)


# In[169]:


df.head()


# In[170]:


#df.drop([col for col, val in df.sum().iteritems() if val == 0], axis=1, inplace=True)


# In[171]:


df.head(1)


# ## Save the processed data in a csv file for each video

# In[172]:

f = path.split('/')[-1].split('.')[0]



parent_dir = "./test_data"
#parent_dir = "/mnt/sde/jagadish/userdata/dl_project/tv_test_data/"

filename = os.path.join(parent_dir + f + '.csv')


df.to_csv(filename, index=True)


# In[173]:


len(df)


# In[ ]:




