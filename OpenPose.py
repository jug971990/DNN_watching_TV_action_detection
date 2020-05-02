#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
from os.path import exists, join, basename, splitext
from sys import argv
import sys



# In[ ]:


git_repo_url = 'https://github.com/CMU-Perceptual-Computing-Lab/openpose.git'
project_name = splitext(basename(git_repo_url))[0]


# In[ ]:


if not exists(project_name):
  # see: https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/949
  # install new CMake becaue of CUDA10
  wget -q https://cmake.org/files/v3.13/cmake-3.13.0-Linux-x86_64.tar.gz
  tar xfz cmake-3.13.0-Linux-x86_64.tar.gz --strip-components=1 -C /usr/local
  # clone openpose
  git clone -q --depth 1 $git_repo_url
  sed -i 's/execute_process(COMMAND git checkout master WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}\/3rdparty\/caffe)/execute_process(COMMAND git checkout f019d0dfe86f49d1140961f8c7dec22130c83154 WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}\/3rdparty\/caffe)/g' openpose/CMakeLists.txt
  # install system dependencies
  apt-get -qq install -y libatlas-base-dev libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler libgflags-dev libgoogle-glog-dev liblmdb-dev opencl-headers ocl-icd-opencl-dev libviennacl-dev
  # install python dependencies
  pip install -q youtube-dl
  # build openpose
  cd openpose && rm -rf build || true && mkdir build && cd build && cmake .. && make -j`nproc`


# In[ ]:


cd /content/openpose


# In[ ]:


path = sys.argv[1]


# In[2]:


filename = path.split('/')[-1].split('.')[0]


# In[3]:


parent_dir = "./test_videos_json_files"


# In[6]:


path1 = os.path.join(parent_dir, filename)
os.makedirs(path1, exist_ok=True)


# In[7]:


path1


# In[ ]:





# In[ ]:


./build/examples/openpose/openpose.bin --video  path --hand  --write_images 'jag_9038_frames' --write_json path1 --disable_blending --display 0


# In[ ]:


get_ipython().system('zip -r /content/openpose/jag_9038.zip /content/openpose/jag_9038')


# In[ ]:


from google.colab import files
files.download("/content/openpose/jag_9038.zip")


# In[ ]:


#!zip -r /content/openpose/jag_2447_frames.zip /content/openpose/jag_2447_frames


# In[ ]:


#from google.colab import files
#files.download("/content/openpose/jag_2447_frames.zip")


# In[ ]:




