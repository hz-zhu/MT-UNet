from hzhu_net import *

import torch, os, copy
import torch.nn as nn
from torch.nn import functional as F

def conv_block(in_ch, out_ch):
    conv = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True))
    return conv

def up_conv(in_ch, out_ch):
    up = nn.Sequential(
        nn.Upsample(scale_factor=2),
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )

    return up


    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out, psi
    
def classification_head(in_features, mid_features, out_features, dropout_rate):
        if mid_features is not None:
            r = nn.Sequential()
            r.add_module('linear_1', nn.Linear(in_features=in_features, out_features=mid_features))
            if dropout_rate is not None:
                if dropout_rate>0.0:
                    r.add_module('dropout', nn.Dropout(p=dropout_rate))
            r.add_module('relu_1', nn.ReLU())
            r.add_module('linear_2', nn.Linear(in_features=mid_features, out_features=out_features))
            return r
        else:
            return nn.Linear(in_features=in_features, out_features=out_features)

class UNet_Chunk(Module):
    def __init__(self, in_channels, filter_list):
        super().__init__()

        self.in_channels = in_channels
        self.filter_list = filter_list

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(self.in_channels, self.filter_list[0])
        self.Conv2 = conv_block(self.filter_list[0], self.filter_list[1])
        self.Conv3 = conv_block(self.filter_list[1], self.filter_list[2])
        self.Conv4 = conv_block(self.filter_list[2], self.filter_list[3])
        self.Conv5 = conv_block(self.filter_list[3], self.filter_list[4])

        self.Up5 = up_conv(self.filter_list[4], self.filter_list[3])
        self.Up_conv5 = conv_block(self.filter_list[4], self.filter_list[3])

        self.Up4 = up_conv(self.filter_list[3], self.filter_list[2])
        self.Up_conv4 = conv_block(self.filter_list[3], self.filter_list[2])

        self.Up3 = up_conv(self.filter_list[2], self.filter_list[1])
        self.Up_conv3 = conv_block(self.filter_list[2], self.filter_list[1])

        self.Up2 = up_conv(self.filter_list[1], self.filter_list[0])
        self.Up_conv2 = conv_block(self.filter_list[1], self.filter_list[0])

    def forward(self, x):
        
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)

        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        return e5, d2
    
class MTL_UNet(UNet_Chunk):
    
    def __init__(self, in_channels, filter_list, out_dict):
        super().__init__(in_channels, filter_list)
        self.out_dict = out_dict
        self.init()
        
    def init(self):
        self.dummy_tensor = nn.Parameter(torch.tensor(0), requires_grad=False)
        
        if self.out_dict is None:
            self.out_conv = nn.Conv2d(self.filter_list[0], 1, kernel_size=1, stride=1, padding=0)
        else:
            if 'class' in self.out_dict:
                if self.out_dict['class']>0:
                    self.out_classification = classification_head(
                        in_features=self.filter_list[0]+self.filter_list[-1],
                        mid_features=self.filter_list[0], out_features=self.out_dict['class'],
                        dropout_rate=0.25)
            if 'image' in self.out_dict:
                if self.out_dict['image']>0:
                    self.out_conv_image = conv_bn_acti_drop(
                        in_channels=self.filter_list[0],
                        out_channels=self.filter_list[0],
                        kernel_size=3,
                        activation=nn.ReLU,
                        normalize=nn.BatchNorm2d,
                        padding=1,
                        dropout_rate=0.0,
                        sequential=None)
                    self.out_conv_image.add_module(
                        'conv_last', nn.Conv2d(self.filter_list[0], self.out_dict['image'], kernel_size=1, stride=1, padding=0))
    
    def forward(self, x):
        e5, d2 = super().forward(x)
        
        if self.out_dict is None:
            y = self.out_conv(d2)
            return self.dummy_tensor, y
        else:
            r = []
            if 'class' in self.out_dict:
                if self.out_dict['class']>0:
                    average_pool_e5 = e5.mean(dim=(-2,-1))
                    average_pool_d2 = d2.mean(dim=(-2,-1))
                    average_pool = torch.cat((average_pool_e5, average_pool_d2), dim=1)
                    y_class = self.out_classification(average_pool)
                    r.append(y_class)
                else:
                    r.append(self.dummy_tensor)
            else:
                r.append(self.dummy_tensor)
                
            if 'image' in self.out_dict:
                if self.out_dict['image']>0:
                    y_image = self.out_conv_image(d2)
                    r.append(y_image)
                else:
                    r.append(self.dummy_tensor)
            else:
                r.append(self.dummy_tensor)
                
            return tuple(r)
        
class MTL_UNet_preset(MTL_UNet):
    
    def __init__(self, device, out_dict, loss_dict):
        self.device = device
        base = 64 if os.getcwd()[0] == '/' else 2
        super().__init__(in_channels=1, filter_list=[base*(2**i) for i in range(5)], out_dict=out_dict)
        
        self.loss_dict = loss_dict
        self.mt_param_init()
        
        self.to(self.device)
        
    def mt_param_init(self):
        if 'class' in self.out_dict:
            if self.out_dict['class']>0:
                if self.loss_dict['class'] is not None:
                    self.lg_sigma_class = nn.Parameter(torch.tensor(self.loss_dict['class'], device=self.device, dtype=torch.float32))
                else:
                    self.lg_sigma_class = torch.tensor(0.0, device=self.device, dtype=torch.float32)
                    
        if 'image' in self.out_dict:
            if self.out_dict['image']>0:
                if not isinstance(self.loss_dict['image'], (list, tuple)):
                    self.loss_dict['image'] = [self.loss_dict['image'],]
                for item in self.loss_dict['image']:
                    if item is not None:
                        self.lg_sigma_image = nn.Parameter(torch.tensor(item, device=self.device, dtype=torch.float32))
                    else:
                        self.lg_sigma_image = torch.tensor(0.0, device=self.device, dtype=torch.float32)
                    
    def compute_loss_class(self, y_pred, y_true, loss_function):
        
        sigma = torch.exp(self.lg_sigma_class)
        loss_raw = loss_function(y_pred, y_true)
        loss_weighted = loss_raw/sigma/sigma+torch.log(sigma+1.0)
        
        return sigma, loss_raw, loss_weighted
    
    def compute_loss_image(self, y_pred, y_true, loss_function, idx):
        
        sigma = torch.exp(self.lg_sigma_image)
        loss_raw = loss_function(y_pred, y_true)
        loss_weighted = loss_raw/sigma/sigma/2.0+torch.log(sigma+1.0)
        
        return sigma, loss_raw, loss_weighted
    
    def compute_loss(self, y_class_pred, y_image_pred, y_class_true, y_image_true, loss_class, loss_image_list):
        
        class_sigma, class_loss_raw, class_loss_weighted = self.compute_loss_class(
            y_pred=y_class_pred, y_true=y_class_true, loss_function=loss_class)
        
        image_sigma, image_loss_raw, image_loss_weighted = self.compute_loss_image(
            y_pred=y_image_pred, y_true=y_image_true, loss_function=loss_image_list[0], idx=0)
        
        loss_sum = class_loss_weighted+image_loss_weighted
        
        r = {'loss_sum':loss_sum,
             'class_loss_raw':class_loss_raw,
             'image_loss_raw':image_loss_raw}
            
        return r
    
    def get_status(self):
        r = []
        r.append(torch.exp(self.lg_sigma_class).detach().clone().cpu())
        r.append(torch.exp(self.lg_sigma_image).detach().clone().cpu())
        return r
    
    def get_status_str(self):
        stats = self.get_status()
        r = ''
        for item in stats:
            r += '%.2e '%item
            
        return r
    