import sys, json

import torch
from torch import nn as nn

import copy
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages

import sklearn.metrics as M
    
class MetricsHandle_Class:
    
    def __init__(self):
        self.data = []
        
    def __len__(self):
        return len(self.data)
    
    def add_data(self, Y, Y_pred):
        
        Y = Y.detach().clone().cpu()
        Y_pred = nn.Softmax(dim=1)(Y_pred.detach().clone().cpu())
        
        batch_num = Y_pred.shape[0]
        
        for i in range(batch_num):
            local_Y = Y[i:i+1]
            local_Y_pred = Y_pred[i,:]
            
            local_data = {}
            local_data['Y'] = local_Y
            local_data['Y_pred'] = local_Y_pred
            
            self.data.append(local_data)
            
    def __getitem__(self, i):
        return self.data[i]
    
    def compute_classification_report(self):
        Y_true = torch.cat([self[i]['Y'] for i in range(len(self))], dim=0).float().numpy()
        Y_true_onehot = one_hot_encoding(Y_true, class_num=3)
        Y_score = torch.stack([self[i]['Y_pred'] for i in range(len(self))], dim=0).numpy()
        Y_pred = np.argmax(Y_score, axis=1)

        self.classification_report = M.classification_report(Y_true, Y_pred, output_dict=True)
        
        for i in range(3):
            class_name = 'class_%d'%(i)
            self.classification_report[class_name+'_fpr'], self.classification_report[class_name+'_tpr'], _ = \
                M.roc_curve(y_score=Y_score[:,i], y_true=Y_true_onehot[:,i])
            self.classification_report[class_name+'_ROC_AUC'] = \
                M.auc(self.classification_report[class_name+'_fpr'], self.classification_report[class_name+'_tpr'])
        
        self.classification_report['micro_fpr'], self.classification_report['micro_tpr'], _ = \
            M.roc_curve(y_score=Y_score.ravel(), y_true=Y_true_onehot.ravel())
        self.classification_report['micro_ROC_AUC'] = \
            M.auc(self.classification_report['micro_fpr'], self.classification_report['micro_tpr'])
        
        for item in self.classification_report:
            if isinstance(self.classification_report[item], np.ndarray):
                self.classification_report[item] = self.classification_report[item].tolist()
    
    def get_evaluation(self):
        if not hasattr(self, 'classification_report'):
            self.compute_classification_report()
        return self.classification_report
    
    def get_key_evaluation(self):
        Y_true = torch.cat([self[i]['Y'] for i in range(len(self))], dim=0).float().numpy()
        Y_score = torch.stack([self[i]['Y_pred'] for i in range(len(self))], dim=0).numpy()
        Y_pred = np.argmax(Y_score, axis=1)
        return M.accuracy_score(Y_true, Y_pred)
                
    def save_outputs(self, name, path):
        r = []
        for i in range(len(self)):
            item = self[i]
            r.append({'Y':item['Y'].tolist(), 'Y_pred':item['Y_pred'].tolist()})
            
        r = pd.DataFrame(r)
        r.to_csv(path+'/'+name+'.csv')
        
    def save_classification_report(self, name, path):
        if not hasattr(self, 'classification_report'):
            self.compute_classification_report()
        with open(path+'/'+name+'.json', 'w') as f:
            json.dump(self.classification_report, f, ensure_ascii=False, indent=4)
            
def one_hot_encoding(x, class_num=None):
    if class_num is None:
        class_num = np.max(x)+1
    r = []
    for item in x:
        tmp = np.zeros((1,class_num), dtype=np.int32)
        tmp[0,int(item)] = 1.0
        r.append(tmp)
    return np.concatenate(r, axis=0)