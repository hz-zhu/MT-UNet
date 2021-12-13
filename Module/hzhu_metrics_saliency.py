import sys, json

import torch
from torch import nn as nn
from torch.nn import functional as F

import copy
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages

import sklearn.metrics as M
import scipy

class MetricsHandle_Saliency:
    
    def __init__(self):
        self.data = []
        
    def __len__(self):
        return len(self.data)
    
    def add_data(self, Y, Y_pred):
        
        #X = X.detach().clone().cpu()
        Y = Y.detach().clone().cpu()
        Y_pred = Y_pred.detach().clone().cpu()
        
        batch_num = Y_pred.shape[0]
        
        for i in range(batch_num):
            local_Y = Y[i,:,:,:]
            local_Y_pred = Y_pred[i,:,:,:]
            
            local_data = {}
            local_data['Y'] = local_Y
            local_data['Y_pred_log'] = local_Y_pred
            local_data['Y_pred'] = torch.exp(local_data['Y_pred_log'])
            
            check_sum = local_data['Y_pred'].sum()
            if check_sum>1.0+1e-3 or check_sum<1.0-1e-3:
                print('Y_pred check sum failed with %e'%check_sum)
                local_data['Y_pred'] /= check_sum
                
            check_sum = local_data['Y'].sum()
            if check_sum>1.0+1e-3 or check_sum<1.0-1e-3:
                print('Y check sum failed with %e'%check_sum)
                local_data['Y'] /= check_sum
            
            self.data.append(local_data)
            
    def __getitem__(self, i):
        return self.data[i]
    
    def compute_prediction_report(self):
        self.KL_loss_list = []
        self.CC_list = []
        
        self.EMD_list = []
        self.histogram_similarity_list = []
        
        with torch.no_grad():
            for item in self.data:
                
                KL_loss = F.kl_div(item['Y_pred_log'], item['Y'], reduction='batchmean')
                self.KL_loss_list.append(KL_loss)
                
                CC, p = scipy.stats.pearsonr(item['Y_pred'].flatten().numpy(), item['Y'].flatten().numpy())
                self.CC_list.append(CC)
                
                HI = torch.minimum(item['Y_pred'], item['Y']).sum()                
                if HI>1.0+1e-5:
                    print('Invalid HI encountered', HI)
                else:
                    self.histogram_similarity_list.append(HI)
            
        self.KL_loss_list = np.array(self.KL_loss_list)
        self.histogram_similarity_list = np.array(self.histogram_similarity_list)
        self.CC_list = np.array(self.CC_list)
        
        self.prediction_report = {
            'KL_mean': float(self.KL_loss_list.mean()),
            'KL_median': float(np.median(self.KL_loss_list)),
            'KL_std': float(self.KL_loss_list.std()),
            
            'CC_mean': float(self.CC_list.mean()),
            'CC_median': float(np.median(self.CC_list)),
            'CC_std': float(self.CC_list.std()),
        
            'HS_mean': float(self.histogram_similarity_list.mean()),
            'HS_median': float(np.median(self.histogram_similarity_list)),
            'HS_std': float(self.histogram_similarity_list.std())}
    
    def get_evaluation(self):
        if not hasattr(self, 'prediction_report'):
            self.compute_prediction_report()
        return self.prediction_report
    
    def get_key_evaluation(self):
        self.metrics_list = []
        
        with torch.no_grad():
            for item in self.data:
                KL_loss = F.kl_div(item['Y_pred_log'], item['Y'], reduction='batchmean')
                self.metrics_list.append(KL_loss)

        self.metrics_list = np.array(self.metrics_list)
        return float(self.metrics_list.mean())
        
    def save_prediction_report(self, name, path):
        if not hasattr(self, 'prediction_report'):
            self.compute_prediction_report()
        with open(path+'/'+name+'.json', 'w') as f:
            json.dump(self.prediction_report, f, ensure_ascii=False, indent=4)