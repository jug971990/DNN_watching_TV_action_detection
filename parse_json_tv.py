#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[3]:


import pandas as pd
import time
import numpy as np
import math
from itertools import islice 
import json
from pandas.io.json import json_normalize
import os


# ## Read and parse json files for each video

# In[9]:




# this finds our json files
path_to_json = "/mnt/sde/jagadish/userdata/dl_project/tv_json_files_new/jagtv_8306/"
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]

# here I define my pandas Dataframe with the columns I want to get from the json
jsons_data = pd.DataFrame(columns=['index','person_id','po', 'hl', 'hr'])
df = pd.DataFrame()
df_all = pd.DataFrame()

# we need both the json and an index number so use enumerate()
for index, js in enumerate(json_files):
    with open(os.path.join(path_to_json, js)) as json_file:
        json_text = json.load(json_file)
        #print(str(json_file).split('/')[-1].split('_')[2])
        # here you need to know the layout of your json and each json has to have
        # the same structure (obviously not the structure I have here)
        index = (str(json_file).split('/')[-1].split('_')[2])
        person_id = json_text['people'][0]['person_id']
        po = json_text['people'][0]['pose_keypoints_2d']
        hl = json_text['people'][0]['hand_left_keypoints_2d']
        hr = json_text['people'][0]['hand_right_keypoints_2d'] 
        #city = json_text['features'][0]['properties']['name']
        #lonlat = json_text['features'][0]['geometry']['coordinates']
        # here I push a list of data into a pandas DataFrame at row given by 'index'
        jsons_data.loc[index] = [index,person_id, po, hl, hr]


# now that we have the pertinent json data in our DataFrame let's look at it
#print(df_all)
print(jsons_data)


# ## Preprocessing data

# In[10]:


df = jsons_data


# In[11]:


df['index'] = df['index'].astype(int)


# In[12]:


df=df.sort_index()


# In[13]:


df.head(5)


# In[14]:


gf = pd.DataFrame()


# In[15]:


gf=df


# In[16]:


gf.head(1)


# In[17]:


gf = gf.join(gf['po'].apply(pd.Series).add_prefix('po'))
gf = gf.join(gf['hl'].apply(pd.Series).add_prefix('hl'))
gf = gf.join(gf['hr'].apply(pd.Series).add_prefix('hr'))


# In[18]:


gf = gf.drop(['po', 'hl', 'hr', 'person_id', 'index'], axis=1)


# In[19]:


gf.head(1)


# In[20]:


gf = gf.drop(gf.iloc[:, 2::3],axis=1)


# In[21]:


gf.head(1)


# In[21]:


mf = pd.DataFrame()


# In[22]:


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


# In[23]:


mf.head(1)


# In[24]:


hf=pd.DataFrame()


# In[25]:


hf = mf


# In[26]:


hf.head(1)


# In[27]:


hf.shape[1]


# ## Euclidean distances between body key points

# In[28]:


def eudis5(v1, v2): # Function to calculate euclidean distance between two points
    dist = [(a - b)**2 for a, b in zip(v1, v2)]
    dist = math.sqrt(sum(dist))
    return dist


# In[29]:


hf['pd'] = ''
hf['pd'] = hf['pd'].apply(list)


# In[30]:


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

# In[34]:


hf['hld'] = ''
hf['hld'] = hf['hld'].apply(list)


# In[36]:


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

# In[37]:


hf['hrd'] = ''
hf['hrd'] = hf['hrd'].apply(list)


# In[39]:


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


# In[40]:


hf.head(5)


# ## Filtering the dataframe with desired columns

# In[41]:


df= pd.DataFrame()


# In[42]:


df=hf.filter(items=['pd',  'hld', 'hrd'])


# In[43]:


df.head(1)


# In[44]:


df = df.join(df['pd'].apply(pd.Series).add_prefix('p_'))
df = df.join(df['hld'].apply(pd.Series).add_prefix('hl_'))
df = df.join(df['hrd'].apply(pd.Series).add_prefix('hr_'))


# In[45]:


df.head(1)


# In[46]:


df = df.join(df['p_0'].apply(pd.Series).add_prefix('pd_'))
df = df.join(df['hl_0'].apply(pd.Series).add_prefix('hld_'))
df = df.join(df['hr_0'].apply(pd.Series).add_prefix('hrd_'))


# In[47]:


df.head(1)


# In[48]:


df = df.drop(['pd', 'hld', 'hrd','p_0', 'hl_0', 'hr_0'], axis=1)


# In[49]:


df.head()


# In[51]:


df.drop([col for col, val in df.sum().iteritems() if val == 0], axis=1, inplace=True)


# In[52]:


df.head(1)


# ## Save the processed data in a csv file for each video

# In[53]:


df.to_csv("/mnt/sde/jagadish/userdata/dl_project/tv_test_data/tv_8361.csv", index=True)

