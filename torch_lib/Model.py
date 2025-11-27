import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def OrientationLoss(orient_batch, orientGT_batch, confGT_batch):

    batch_size = orient_batch.size()[0]
    indexes = torch.max(confGT_batch, dim=1)[1]

    # extract just the important bin
    orientGT_batch = orientGT_batch[torch.arange(batch_size), indexes]
    orient_batch = orient_batch[torch.arange(batch_size), indexes]

    theta_diff = torch.atan2(orientGT_batch[:,1], orientGT_batch[:,0])
    estimated_theta_diff = torch.atan2(orient_batch[:,1], orient_batch[:,0])

    return -1 * torch.cos(theta_diff - estimated_theta_diff).mean()

class Model(nn.Module):
    def __init__(self, features=None, bins=2, w = 0.4):
        super(Model, self).__init__()
        self.bins = bins
        self.w = w
        self.features = features
        dummy_input = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            dummy_output = self.features(dummy_input)
        flatten_dim = dummy_output.reshape(1, -1).shape[1]
        self.orientation = nn.Sequential(
                    nn.Linear(flatten_dim, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, bins*2) # to get sin and cos
                )
        self.confidence = nn.Sequential(
                    nn.Linear(flatten_dim, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, bins),
                    # nn.Softmax()
                    #nn.Sigmoid()
                )
        self.dimension = nn.Sequential(
                    nn.Linear(flatten_dim, 512),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(512, 512),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(512, 3)
                )
        
        """ modifier but doesn't work well
        self.orient_dim = bins * 2
        self.proj_orient = nn.Linear(flatten_dim, self.orient_dim) 
        self.mod_orient = SelfModifier(dim=self.orient_dim)

        self.conf_dim = bins
        self.proj_conf = nn.Linear(flatten_dim, self.conf_dim)
        self.mod_conf = SelfModifier(dim=self.conf_dim)

        self.dim_dim = 3
        self.proj_dim = nn.Linear(flatten_dim, self.dim_dim)
        self.mod_dim = SelfModifier(dim=self.dim_dim)
        """
    def forward(self, x):
        x = self.features(x) 
        
        x = x.reshape(x.size(0), -1)
        feat_vec = x
        orientation = self.orientation(x)
        orientation_shaped = orientation.reshape(-1, self.bins, 2)
        orientation_norm = F.normalize(orientation_shaped, dim=2)
        confidence = self.confidence(x)
        dimension = self.dimension(x)
        return orientation_norm, confidence, dimension, feat_vec, orientation
    
    '''
    def get_modifiers(self, feature, raw_orient, err_orient, raw_conf, err_conf, raw_dim, err_dim):
        
        k_orient = self.proj_orient(feature)
        mod_orient = self.mod_orient(key=k_orient, value=raw_orient, error_signal=err_orient)

        k_conf = self.proj_conf(feature)
        mod_conf = self.mod_conf(key=k_conf, value=raw_conf, error_signal=err_conf)

        k_dim = self.proj_dim(feature)
        mod_dim = self.mod_dim(key=k_dim, value=raw_dim, error_signal=err_dim)

        return mod_orient, mod_conf, mod_dim
    
    def compute_nested_loss(self, feat, raw_orient, raw_conf, raw_dim, gt_orient, gt_conf_idxs, gt_dim):
        # Error Signals
        gt_orient_flat = gt_orient.view(raw_orient.shape)
        err_orient = (gt_orient_flat - raw_orient).detach()
        
        gt_conf_hot = F.one_hot(gt_conf_idxs, num_classes=self.bins).float()
        err_conf = (gt_conf_hot - F.softmax(raw_conf, dim=1)).detach()
        
        err_dim = (gt_dim - raw_dim).detach()

        # Modifiers
        mod_orient = self.mod_orient(self.proj_orient(feat), raw_orient.detach(), err_orient)
        mod_conf = self.mod_conf(self.proj_conf(feat), raw_conf.detach(), err_conf)
        mod_dim = self.mod_dim(self.proj_dim(feat), raw_dim.detach(), err_dim)

        # Loss
        # Head -> GT + Mod
        loss_in = (F.mse_loss(raw_orient, gt_orient_flat + mod_orient.detach()) +
                   F.mse_loss(raw_conf, (gt_conf_hot * 10) + mod_conf.detach()) +
                   F.mse_loss(raw_dim, gt_dim + mod_dim.detach()))
        
        # Head.detach + Mod -> GT
        loss_out = (F.mse_loss(raw_orient.detach() + mod_orient, gt_orient_flat) +
                    F.mse_loss(raw_conf.detach() + mod_conf, gt_conf_hot * 10) +
                    F.mse_loss(raw_dim.detach() + mod_dim, gt_dim))

        return 0.1 * (loss_in + loss_out) 
    '''
    def prepare_term_params(self, lr = 1e-4, period_scale = 4):
        term_1 = list(self.features[:23].parameters())  # fast
        term_2 = list(self.features[23:].parameters())  # medium
        term_3 = list(self.orientation.parameters()) + \
                 list(self.confidence.parameters()) + \
                 list(self.dimension.parameters())  # slowest
                 
        #term_3 = list(self.mod_orient.parameters()) + list(self.proj_orient.parameters()) + \
        #         list(self.mod_conf.parameters()) + list(self.proj_conf.parameters()) + \
        #         list(self.mod_dim.parameters()) + list(self.proj_dim.parameters()) #modifiers
        #output form
        """
        [{'params': terms['term_1'], 'lr': lr, 'period': 1}
        .
        .
        .
        {'params': terms['term_n'], 'lr': 1e-4, 'period': period_scale^n}]
        """
        params_list = []
        params_list.append({'params': term_1, 'lr': lr, 'period': 1})
        params_list.append({'params': term_2, 'lr': lr*np.sqrt(0.5), 'period': period_scale})
        params_list.append({'params': term_3, 'lr': lr*0.5, 'period': period_scale**2})
        return params_list
    
class SelfModifier(nn.Module):
    def __init__(self, dim: int, hidden_multiplier: int = 4):
        super().__init__()
        hidden = dim * hidden_multiplier
        self.net = nn.Sequential(
            nn.Linear(dim * 3, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, dim),
        )
    
    def forward(self, key, value, error_signal):
        return self.net(torch.cat([key, value, error_signal], dim=-1))