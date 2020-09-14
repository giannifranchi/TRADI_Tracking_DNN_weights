# imports
import matplotlib.pyplot as plt
import numpy as np
#import gpytorch
import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim





class ThreehiddedLayersNet_fixed(torch.nn.Module):
    def __init__(self, D_in,H1,H2,H3, D_out,p=0.5):

        super(ThreehiddedLayersNet_fixed, self).__init__()
        self.D_in=D_in
        self.H1 = H1
        self.H2 = H2
        self.H3 = H3
        self.D_out = D_out

        self.fc1 = torch.nn.Linear(self.D_in, self.H1)#,bias=False)
        self.bn1 = nn.BatchNorm1d(num_features=H1)
        self.fc2 = torch.nn.Linear(self.H1, self.H2)#,bias=False)
        self.bn2 = nn.BatchNorm1d(num_features=H2)
        self.fc3 = torch.nn.Linear(self.H2, self.H3)#,bias=False)
        self.bn3 = nn.BatchNorm1d(num_features=H3)
        self.fpred = torch.nn.Linear(self.H3, self.D_out)#,bias=False)
        self.p = p
        torch.nn.init.kaiming_normal_(self.fc1.weight, a=0, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.fc2.weight, a=0, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.fc3.weight, a=0, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.fpred.weight, a=0, mode='fan_in', nonlinearity='relu')

        nn.init.constant_(self.fc1.bias,0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.fc3.bias, 0)
        nn.init.constant_(self.fpred.bias, 0)
    def forward(self, x):

        h1 = F.relu(self.bn1(self.fc1(x)))
        #h_relu = F.dropout(h1, p=self.p, training=self.training)
        h_relu = F.relu(self.bn2(self.fc2(h1)))
        h_relu = F.dropout(h_relu, p=self.p, training=self.training)
        h_relu = F.relu(self.bn3(self.fc3(h_relu)))
        h_relu = F.dropout(h_relu, p=self.p, training=self.training)
        y_pred = self.fpred(h_relu)
        return y_pred



class ThreehiddedLayersNet_fixed2(torch.nn.Module):
    def __init__(self, D_in,H1,H2,H3, D_out,p=0.5):

        super(ThreehiddedLayersNet_fixed2, self).__init__()
        self.D_in=D_in
        self.H1 = H1
        self.H2 = H2
        self.H3 = H3
        self.D_out = D_out

        self.fc1 = torch.nn.Linear(self.D_in, self.H1)#,bias=False)
        self.bn1 = nn.BatchNorm1d(num_features=H1)
        self.fc2 = torch.nn.Linear(self.H1, self.H2)#,bias=False)
        self.bn2 = nn.BatchNorm1d(num_features=H2)
        self.fc3 = torch.nn.Linear(self.H2, self.H3)#,bias=False)
        self.bn3 = nn.BatchNorm1d(num_features=H3)
        self.fpred = torch.nn.Linear(self.H3, self.D_out)#,bias=False)
        self.p = p
        torch.nn.init.kaiming_normal_(self.fc1.weight, a=0, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.fc2.weight, a=0, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.fc3.weight, a=0, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.fpred.weight, a=0, mode='fan_in', nonlinearity='relu')

        nn.init.constant_(self.fc1.bias,0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.fc3.bias, 0)
        nn.init.constant_(self.fpred.bias, 0)
    def forward(self, x):

        x = x.view(x.shape[0], -1) # This reshape is needed for the BN update
        h1 = F.relu(self.bn1(self.fc1(x)))
        #h_relu = F.dropout(h1, p=self.p, training=self.training)
        h_relu = F.relu(self.bn2(self.fc2(h1)))
        h_relu = F.dropout(h_relu, p=self.p, training=self.training)
        h_relu = F.relu(self.bn3(self.fc3(h_relu)))
        h_relu = F.dropout(h_relu, p=self.p, training=self.training)
        y_pred = self.fpred(h_relu)
        return y_pred

