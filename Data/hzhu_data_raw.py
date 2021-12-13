import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
import torch, json, pickle, copy

from torch.nn import functional as F

from hzhu_gen import *
from pydicom import dcmread
from scipy.sparse import coo_matrix

from datetime import datetime

class GazeDataHandle:
    
    def __init__(self, root_path):
        self.root_path = root_path
        self.master_sheet = pd.read_csv(self.root_path+'/'+'master_sheet.csv')
        self.eye_gaze = pd.read_csv(self.root_path+'/'+'eye_gaze.csv')
        
        self.selected_df = copy.deepcopy(self.master_sheet)
        self.selected_DICOM_ID = self.selected_df['dicom_id'].tolist()
        
        self.process()
        
    def process(self):
        self.groups = self.eye_gaze.groupby(['DICOM_ID'])
        self.data_gaze = {item[0]:item[1] for item in self.groups if len(item[0])==44 and item[0] in self.selected_DICOM_ID}

    def __getitem__(self, i):
        ID = self.selected_DICOM_ID[i]
        info = self.selected_df[self.selected_df['dicom_id']==ID]
        gaze = self.data_gaze[ID]
        
        return {'info':info, 'gaze_raw':gaze}
    
    def __len__(self):
        return len(self.selected_DICOM_ID)
    
class CXRDataHandle:
    
    def __init__(self, root_path):
        self.root_path = root_path
    
    def __getitem__(self, info):
        path = info['path'].iloc[0][10:]
        image = dcmread(self.root_path+'/'+path).pixel_array.astype(np.float32)
        return image
    
class SegDataHandle:
    
    def __init__(self, root_path):
        self.root_path = root_path
        
    def __getitem__(self, info):
        ID = info['dicom_id'].iloc[0]
        left_lung = cv.imread(self.root_path+'/audio_segmentation_transcripts/'+ID+'/left_lung.png', cv.IMREAD_UNCHANGED).astype(np.bool_)
        mediastanum = cv.imread(self.root_path+'/audio_segmentation_transcripts/'+ID+'/mediastanum.png', cv.IMREAD_UNCHANGED).astype(np.bool_)
        right_lung = cv.imread(self.root_path+'/audio_segmentation_transcripts/'+ID+'/right_lung.png', cv.IMREAD_UNCHANGED).astype(np.bool_)
        
        return {'lung':(left_lung+right_lung).astype(np.bool_), 'heart':mediastanum}
    
class MasterDataHandle:
    
    def __init__(self, gaze_path, cxr_path, blur):
        self.blur = blur
        self.gaze_path = gaze_path
        self.cxr_path = cxr_path
        self.gazeData = GazeDataHandle(self.gaze_path)
        self.segData = SegDataHandle(self.gaze_path)
        self.cxrData = CXRDataHandle(self.cxr_path)
        
    def __len__(self):
        return len(self.gazeData)
    
    def __getitem__(self, i):
        r = self.gazeData[i]
        r['cxr'] = self.cxrData[r['info']]
        seg = self.segData[r['info']]
        r['lung'] = seg['lung']
        r['heart'] = seg['heart']
        r['gaze'] = get_gaze_heatmap(r, self.blur)
        
        if r['info']['Normal'].iloc[0]==1 and r['info']['CHF'].iloc[0]==0 and r['info']['pneumonia'].iloc[0]==0:
            r['Y'] = 0
        elif r['info']['Normal'].iloc[0]==0 and r['info']['CHF'].iloc[0]==1 and r['info']['pneumonia'].iloc[0]==0:
            r['Y'] = 1
        elif r['info']['Normal'].iloc[0]==0 and r['info']['CHF'].iloc[0]==0 and r['info']['pneumonia'].iloc[0]==1:
            r['Y'] = 2
        else:
            assert False
        
        return r
    
    def plot(self, i, blur):
        data = self[i]
        self.process(data, downsample=5)
        
        plt.figure(figsize=(10,8))
        plt.subplot(2,3,1)
        plt.imshow(data['cxr'])
        plt.title(data['Y'])
        
        plt.subplot(2,3,2)
        plt.imshow(data['gaze'])
        plt.title(data['gaze'].shape)
        
        plt.subplot(2,3,3)
        plt.imshow(data['gaze']/torch.max(data['gaze'])+data['cxr']/torch.max(data['cxr']))
        
        plt.subplot(2,3,4)
        plt.imshow(data['heart'])
        
        plt.subplot(2,3,5)
        plt.imshow(data['lung'])
        
        
            
    def save(self, data, path, downsample):        
        local_name = '%s'%data['info']['dicom_id'].iloc[0]
        
        item = ['gaze', 'cxr', 'Y', 'heart', 'lung']
        r = {}
        for key in item:
            r[key] = data[key]
            
        torch.save(r, path+'/'+local_name+'.pt')
    
    def process(self, data, downsample):
       
        data['gaze'] = torch.tensor(data['gaze'], requires_grad=False, dtype=torch.float32)
        data['cxr'] = torch.tensor(data['cxr'], requires_grad=False, dtype=torch.float32)
        data['lung'] = torch.tensor(data['lung'], requires_grad=False, dtype=torch.bool)
        data['heart'] = torch.tensor(data['heart'], requires_grad=False, dtype=torch.bool)
        data['Y'] = torch.tensor(data['Y'], requires_grad=False, dtype=torch.int8)
        
        shape = data['cxr'].shape
        if shape[0]<=shape[1]:
            torch.transpose(data['cxr'], 0, 1)
            torch.transpose(data['gaze'], 0, 1)
            torch.transpose(data['lung'], 0, 1)
            torch.transpose(data['heart'], 0, 1)

        shape = data['cxr'].shape
        H = (int(3056/downsample/32)+1)*32*downsample
        W = (int(2544/downsample/32)+1)*32*downsample
        if shape[0]<=H or shape[1]<=W:
            padding_left = int((W-shape[1])/2)
            padding_right = W-shape[1]-padding_left
            padding_top = int((H-shape[0])/2)
            padding_bottom = H-shape[0]-padding_top
            data['cxr'] = F.pad(data['cxr'], (padding_left, padding_right, padding_top, padding_bottom))
            data['gaze'] = F.pad(data['gaze'], (padding_left, padding_right, padding_top, padding_bottom))
            data['lung'] = F.pad(data['lung'], (padding_left, padding_right, padding_top, padding_bottom))
            data['heart'] = F.pad(data['heart'], (padding_left, padding_right, padding_top, padding_bottom))
            
        data['gaze'] = data['gaze'][0:H+1:downsample,0:W+1:downsample]
        data['gaze'] -= data['gaze'].min()
        data['gaze'] /= data['gaze'].max()
        
        data['cxr'] = data['cxr'][0:H+1:downsample,0:W+1:downsample]
        data['cxr'] /= data['cxr'].max()
        
        data['lung'] = data['lung'][0:H+1:downsample,0:W+1:downsample]
        data['heart'] = data['heart'][0:H+1:downsample,0:W+1:downsample]
        
    def save_all(self, root_path, downsample, fraction, seed):
        folders = ['Test','Valid','Train']
        counter_full = {item:{i:0 for i in range(3)} for item in folders}
        counter = {i:0 for i in range(3)}
        folder_path = [root_path+'/'+item for item in folders]
        
        for item in folder_path:
            create_folder(item)
            
        N = int(len(self)*fraction)
        index_use, index_n = torch.utils.data.random_split(
            range(len(self)), [N, len(self)-N], generator=torch.Generator().manual_seed(seed))
        
        for idx, i in enumerate(index_use):
            
            data = self[i]
            self.process(data=data, downsample=downsample)
            Y = int(data['Y'])
            counter[Y] += 1
            select = counter[Y]%10
            if select<=1:
                self.save(data=data, path=folder_path[0], downsample=downsample)
                counter_full[folders[0]][Y] += 1
            elif select==2:
                self.save(data=data, path=folder_path[1], downsample=downsample)
                counter_full[folders[1]][Y] += 1
            else:
                self.save(data=data, path=folder_path[2], downsample=downsample)
                counter_full[folders[2]][Y] += 1
                
            if idx%50==0:
                print(idx, '%f%%'%((idx+1)/len(index_use)*100))
                print("-- Current Time =", datetime.now())
        
        disp(counter_full)
        
        
def get_gaze_dot(data):
    X = data['gaze_raw']['X_ORIGINAL'].to_numpy(dtype=np.int32, copy=True)
    Y = data['gaze_raw']['Y_ORIGINAL'].to_numpy(dtype=np.int32, copy=True)
    shape = data['cxr'].shape
    JX = np.logical_and(X>=0, X<shape[1])
    JY = np.logical_and(Y>=0, Y<shape[0])
    J = np.logical_and(JX, JY)

    X = X[J].tolist()
    Y = Y[J].tolist()
    
    r = coo_matrix((np.ones(len(X)), (Y, X)), shape=shape).toarray()
    if np.max(r)>=127:
        print('overflow @ get_gaze_dot(data)')
    r = r.astype(np.int8)
    
    return r

def get_gaze_heatmap(data, blur):
    blur = blur+1 if blur%2==0 else blur
    r = get_gaze_dot(data).astype(np.float32)
    r = cv.GaussianBlur(r, (blur, blur), 0, 0)
    r = r/np.max(r)
    return r