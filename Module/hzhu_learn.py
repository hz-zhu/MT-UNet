import os
import pandas as pd
import numpy as np
import PIL
import matplotlib.pyplot as plt
import matplotlib
import json
matplotlib.use('Agg')

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch import optim

from hzhu_gen import *
from hzhu_data import *
from hzhu_metrics_class import *

class NetLearn:
    
    def __init__(
        self,
        net,
        dataAll,
        criterion,
        optimizer_dict,
        lr,
        lr_min,
        lr_factor,
        epoch_max,
        duration_max,
        patience_reduce_lr,
        patience_early_stop,
        device,
        metrics,
        name,
        path):
        
        self.quickTimer = QuickTimer()
        self.net = net
        self.dataAll = dataAll
        
        self.optimizer_dict = optimizer_dict
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.duration_max = duration_max
        self.epoch_max = epoch_max
        self.criterion = criterion

        self.device = device
        self.patience_reduce_lr = patience_reduce_lr
        self.patience_early_stop = patience_early_stop
        
        self.train_loss_list = []
        self.valid_loss_list = []
        self.test_loss_list = []
        self.metrics_list = []
        self.lr_list = []
        
        self.name = name
        self.path = path
        self.ID = self.name+'_'+random_str()
        self.epoch = 0
        
        self.metrics = metrics
        
        self.set_optimizer()
        self.set_scheduler()
        
        self.model_name = 'NET.pt'
        self.optim_name = 'OPT.pt'
        self.sched_name = 'SCH.pt'
        
        self.create_save_path()
        
        print('ID:', self.ID)
        
    def set_optimizer(self):
        self.optimizer = self.optimizer_dict['optimizer'](self.net.parameters(), lr=self.lr, **self.optimizer_dict['param'])
        
    def set_scheduler(self):
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.lr_factor,
            patience=self.patience_reduce_lr,
            eps=0,
            verbose=False)
        
    def train_iterate(self, dataLoader):
        self.epoch += 1
        self.net.train()
        loss_list = []
    
        for data in dataLoader:
            X = data['cxr'].to(self.device).unsqueeze(1)
            
            Y_class = data['Y'].to(self.device).long()
            Y_saliency = data['gaze'].to(self.device).unsqueeze(1)
            Y_saliency = Y_saliency/Y_saliency.sum(dim=(-2,-1), keepdim=True)

            self.optimizer.zero_grad()
            Y_class_pred, Y_saliency_pred = self.net(X)
            Y_saliency_pred_shape = Y_saliency_pred.shape
            Y_saliency_pred = F.log_softmax(Y_saliency_pred.flatten(start_dim=-2, end_dim=-1), dim=-1).reshape(Y_saliency_pred_shape)
            
            net_list = self.net.compute_loss(
                y_class_pred=Y_class_pred,
                y_image_pred=Y_saliency_pred,
                y_class_true=Y_class, 
                y_image_true=Y_saliency,
                loss_class=self.criterion['class'],
                loss_image_list=[self.criterion['saliency'],])
            
            loss = net_list['loss_sum']
            loss.backward()

            self.optimizer.step()
            loss_list.append(loss.detach().clone().cpu())
            
            del data, X, Y_class, Y_saliency, Y_class_pred, Y_saliency_pred, net_list, loss
        
        return loss_list
    
    def eval_iterate(self, dataLoader):
        self.net.eval()
        loss_list = []
        
        metrics_class = self.metrics['class']()
        metrics_saliency = self.metrics['saliency']()

        with torch.no_grad():
            for data in dataLoader:
                X = data['cxr'].to(self.device).unsqueeze(1)
                
                Y_class = data['Y'].to(self.device).long()
                Y_saliency = data['gaze'].to(self.device).unsqueeze(1)
                Y_saliency = Y_saliency/Y_saliency.sum(dim=(-2,-1), keepdim=True)

                Y_class_pred, Y_saliency_pred = self.net(X)
                Y_saliency_pred_shape = Y_saliency_pred.shape
                Y_saliency_pred = F.log_softmax(Y_saliency_pred.flatten(start_dim=-2, end_dim=-1), dim=-1).reshape(Y_saliency_pred_shape)

                net_list = self.net.compute_loss(
                    y_class_pred=Y_class_pred,
                    y_image_pred=Y_saliency_pred,
                    y_class_true=Y_class, 
                    y_image_true=Y_saliency,
                    loss_class=self.criterion['class'],
                    loss_image_list=[self.criterion['saliency'],])
                
                metrics_class.add_data(Y_class, Y_class_pred)
                metrics_saliency.add_data(Y=Y_saliency, Y_pred=Y_saliency_pred)

                for item in net_list:
                    tmp = {}
                    tmp[item] = net_list[item].detach().clone().cpu()
                    loss_list.append(tmp)

                del data, X, Y_class, Y_saliency, Y_class_pred, Y_saliency_pred, net_list

            return {'metrics_class':metrics_class, 'metrics_saliency':metrics_saliency, 'loss':pd.DataFrame(loss_list)}
    
    
    def save_net(self, path):
        torch.save(self.net.state_dict(), path+'/'+self.model_name)
        torch.save(self.optimizer.state_dict(), path+'/'+self.optim_name)
        torch.save(self.scheduler.state_dict(), path+'/'+self.sched_name)
    
    def load_net(self, path):
        self.net.load_state_dict(torch.load(path+'/'+self.model_name))
        self.net.eval()
        self.optimizer.load_state_dict(torch.load(path+'/'+self.optim_name))
        for pg in self.optimizer.param_groups:
            if len(self.lr_list)>0:
                pg['lr'] = self.lr_list[-1]
        self.scheduler.load_state_dict(torch.load(path+'/'+self.sched_name))
        
    def create_save_path(self):
        self.save_path = self.path+'/'+self.ID
        
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        else:
            print('Training folder already exists!')
        
    def train(self):
        self.valid_loss_min = -np.Inf
        self.epoch_best = 0
        while self.epoch<self.epoch_max:
            time0 = time.perf_counter()
            train_loss = self.train_iterate(self.dataAll('Train'))
            eval_valid = self.eval_iterate(self.dataAll('Valid'))

            self.train_loss_list.append(np.mean(train_loss))
            self.valid_loss_list.append(eval_valid['loss']['loss_sum'].mean())

            eval_metrics = eval_valid['metrics_class'].get_key_evaluation()
            self.metrics_list.append(eval_metrics)

            self.scheduler.step(self.valid_loss_list[-1])
            self.lr_list.append(self.optimizer.param_groups[0]['lr'])

            print('Epoch:%4d, loss_train:%.6f, loss_val:%.6f [%.4e %.4e (%s)], val_acc:%.3f, val_KL:%.3f, time:%3.2fsec'%(\
                    self.epoch,\
                    self.train_loss_list[-1],\
                    self.valid_loss_list[-1],\
                    eval_valid['loss']['class_loss_raw'].mean(),\
                    eval_valid['loss']['image_loss_raw'].mean(),\
                    self.net.get_status_str(),\
                    self.metrics_list[-1],\
                    eval_valid['metrics_saliency'].get_key_evaluation(),\
                    time.perf_counter()-time0))

            del eval_valid
            del eval_metrics
            del train_loss

            if self.train_judgetment():
                break

        self.load_net(self.save_path)
        self.save_training_process()
                
    def train_judgetment(self):
        
        if self.epoch==1:
            self.save_net(self.save_path)
            self.valid_loss_min = self.valid_loss_list[-1]
        else:
            if self.valid_loss_list[-1]<self.valid_loss_min:
                self.valid_loss_min = self.valid_loss_list[-1]
                self.save_net(self.save_path)
                self.epoch_best = self.epoch
                print('- Better network saved')
                
            if self.lr_list[-1]<self.lr_list[-2]:
                print('- Learning rate reduced to %e'%self.lr_list[-1])
                self.load_net(self.save_path)
                print('- Currently best network from epoch %4d reloaded'%self.epoch_best)
                
        if self.lr_list[-1]<self.lr_min:
            print('- Early stopping: learning rate dropped below threshold at %E'%self.lr_min)
            return True
        
        if self.epoch-self.epoch_best>=self.patience_early_stop:
            print('- Early stopping: max non-improving epoch reached at %d'%(self.epoch-self.epoch_best))
            return True
        
        if self.quickTimer()>=self.duration_max:
            print('- Early stopping: Max duration reached %f>=%f (sec)'%(self.quickTimer(), self.duration_max))
            return True
    
    def save_training_process(self):
        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(self.train_loss_list)
        plt.plot(self.valid_loss_list)
        plt.title('loss')

        plt.subplot(3,1,2)
        plt.plot(self.lr_list)
        plt.title('learning rate')

        plt.subplot(3,1,3)
        plt.plot(self.metrics_list)
        plt.title('metrics')
        plt.savefig(self.save_path+'/training_process.png')
        
        plt.close()
        
    def remove_saved_net(self):
        if not hasattr(self, 'model_name'):
            print("The net file does not exist")
            return
        
        if not hasattr(self, 'save_path'):
            print("The net file does not exist")
            return
        
        if os.path.exists(self.save_path+'/'+self.model_name):
            os.remove(self.save_path+'/'+self.model_name)
            print("Saved network file deleted successfully")
        else:
            print("The net file does not exist")
            
    def remove_saved_optim(self):
        if not hasattr(self, 'optim_name'):
            print("The optim file does not exist")
            return
        
        if not hasattr(self, 'save_path'):
            print("The optim file does not exist")
            return
        
        if os.path.exists(self.save_path+'/'+self.optim_name):
            os.remove(self.save_path+'/'+self.optim_name)
            print("Saved optim file deleted successfully")
        else:
            print("The optim file does not exist")
            
    def remove_saved_sched(self):
        if not hasattr(self, 'sched_name'):
            print("The sched file does not exist")
            return
        
        if not hasattr(self, 'save_path'):
            print("The sched file does not exist")
            return
        
        if os.path.exists(self.save_path+'/'+self.sched_name):
            os.remove(self.save_path+'/'+self.sched_name)
            print("Saved sched file deleted successfully")
        else:
            print("The sched file does not exist")
            
    def remove_saved(self):
        self.remove_saved_net()
        self.remove_saved_optim()
        self.remove_saved_sched()
            
    def save_params(self, name, path):
        
        attr_list = [attr for attr in dir(self) if isinstance(getattr(self, attr), (list, tuple, dict, int, float, bool)) and not attr.startswith("_")]
        content = {}
        for attr in attr_list:
            content[attr] = getattr(self, attr)
                
        with open(path+'/'+'params_%s_'%(self.__class__.__name__)+name+'.txt', 'w') as file:
            try:
                json.dump(content, file, indent=4)
            except:
                print('Exception occured at hzhu_learn::NetLearn.save_params(..): content cannot be dumped!')
                file.write(str(content))
        
    def evaluate(self):
        
        eval_test = self.eval_iterate(self.dataAll('Test'))
        eval_test['metrics_class'].compute_classification_report()
        eval_test['metrics_class'].save_classification_report('classification_report', self.save_path)
        eval_test['metrics_class'].save_outputs('classification_results', self.save_path)
        
        r = {key:eval_test['metrics_class'].classification_report[key]\
             for key in eval_test['metrics_class'].classification_report if 'ROC_AUC' in key}
        r['accuracy'] = eval_test['metrics_class'].classification_report['accuracy']
        
        eval_test['metrics_saliency'].compute_prediction_report()
        eval_test['metrics_saliency'].save_prediction_report('prediction_report', self.save_path)
        #eval_test['metrics'].save_outputs('prediction_results', self.save_path)
        
        r = {**eval_test['metrics_saliency'].prediction_report, **r}
        
        return json.dumps(r, indent=4)
    
def index_expand(idx, image, n):
    a, b = idx
    r = []
    for i in range(a-n,a+n+1):
        for j in range(b-n,b+n+1):
            if i>=0 and i<image.shape[0] and j>=0 and j<image.shape[1]:
                r.append((i,j))
    return tuple(r)

def param_select(idx, param_pool):
    if not isinstance(param_pool, (list, tuple)):
        assert False, 'input param_pool type error @hzhu_learn::param_select(idx, param_pool)'
    N = 1
    for item in param_pool:
        if not isinstance(item, (list, tuple)):
            assert False, 'input param_pool content type error @hzhu_learn::param_select(idx, param_pool)'
        N *= len(item)
    n = len(param_pool)
    shape = tuple([len(item) for item in param_pool])
    idx_list = np.unravel_index(range(N), shape)
    
    values = []
    for i in range(n):
        value = param_pool[i][idx_list[i][idx%idx_list[i].shape[0]]]
        values.append(value)
    
    if len(values)==1: return values[0]
    else: return values