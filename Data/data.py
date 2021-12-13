#!/usr/bin/env python
# coding: utf-8

# In[1]:


from hzhu_data_raw import *
import os
from hzhu_gen import *
import argparse


# In[2]:


if __name__ == '__main__':
    QH = QuickHelper()
    
    print('Running on my laptop')
    gaze_path = 'D:/Gaze Dataset/Eye gaze data for chest X-rays/extracted'
    cxr_path = 'D:/Gaze Dataset/MIMIC-CXR & GAZE (master)/RAW/CXR'
    save_path = os.getcwd()
    fraction = 0.01
    
    print('Data preparation completed')
    print(QH)

    downsample = 5
    blur = 500
    path_str = 'data'

    local_save_path = save_path+'/'+path_str
    create_folder(local_save_path)

    DATA = MasterDataHandle(gaze_path=gaze_path, cxr_path=cxr_path, blur=blur)
    DATA.save_all(root_path=local_save_path, downsample=downsample, fraction=fraction, seed=0)

    print('%s generation completed'%path_str)
    print(QH)

