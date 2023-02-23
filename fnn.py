import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class FLC(nn.Module):
    def __init__(self, train_feat, h1=None, h2=None, out=1, doub=False):
        super(FLC, self).__init__()
        self.h1 = h1
        self.h2 = h2
        self.out = out
        if h1 is not None:
            self.fc1 = nn.Linear(train_feat, h1)
            if h2 is not None:
                self.fc2 = nn.Linear(h1, h2)
                self.fc3 = nn.Linear(h2, out)
            else:
                self.fc2 = nn.Linear(h1, out)
        else:
            self.fc1 = nn.Linear(train_feat, out)
        self.double = doub

    def forward(self, x):
        if type(x).__module__ == 'numpy':
            x = torch.from_numpy(x)
        if self.double:
            x = x.double()

        x = F.relu(self.fc1(x))
        if self.h1 is not None:
            x = F.relu(self.fc2(x))
            if self.h2 is not None:
                x = F.relu(self.fc3(x))
        return x


class FLN(nn.Module):
    def __init__(self, train_feat, nc=4, classes=5, h1=None, h2=None, doub=False):
        super(FLN, self).__init__()
        """
        self.fl0 = FL0(cur_host=True) if cur_host == 0 else FL0(cur_host=False)
        self.fl1 = FL1(cur_host=True) if cur_host == 1 else FL1(cur_host=False)
        self.fl2 = FL2(cur_host=True) if cur_host == 2 else FL2(cur_host=False)
        self.fl3 = FL3(cur_host=True) if cur_host == 3 else FL3(cur_host=False)
        """
        self.nc = nc
        self.train_feat = train_feat
        if nc == 2:
            self.fl0, self.fl1 = FLC(train_feat[0], h1=h1, h2=h2, out=classes, doub=doub),\
                                 FLC(train_feat[1], h1=h1, h2=h2, out=classes, doub=doub)
        elif nc == 4:
            self.fl0, self.fl1, self.fl2, self.fl3 = FLC(train_feat[0], h1=h1, h2=h2, out=classes, doub=doub),\
                                                     FLC(train_feat[1], h1=h1, h2=h2, out=classes, doub=doub),\
                                                     FLC(train_feat[2], h1=h1, h2=h2, out=classes, doub=doub),\
                                                     FLC(train_feat[3], h1=h1, h2=h2, out=classes, doub=doub)
        else:
            raise Exception('Invalid number of inputs.')
        # self.fc3 = nn.Linear(4, 5)
        self.fcf = nn.Linear(nc*classes, classes)
        self.v = torch.zeros(nc, classes)  # fill-in
        self.S = torch.zeros(nc)  # set of clients

    def forward(self, x):

        if len(x) == self.nc:
            if self.nc == 2:
                x0, x1 = x
            elif self.nc == 4:
                x0, x1, x2, x3 = x
            else:
                raise Exception('Invalid number of inputs.')
        else:
            tf = np.cumsum(self.train_feat)
            if self.nc == 2:
                x0, x1 = x[:, :tf[0]], x[:, tf[0]:tf[1]]
            elif self.nc == 4:
                x0, x1, x2, x3 = x[:, :tf[0]], x[:, tf[0]:tf[1]], x[:, tf[1]:tf[2]], x[:, tf[2]:]
            else:
                raise Exception('Invalid number of inputs.')
        fl0 = self.fl0(x0) if self.S[0] else self.v[0].repeat(x0.shape[0], 1)
        fl1 = self.fl1(x1) if self.S[1] else self.v[1].repeat(x1.shape[0], 1)
        if self.nc >= 3:
            fl2 = self.fl2(x2) if self.S[2] else self.v[2].repeat(x2.shape[0], 1)
        if self.nc >= 4:
            fl3 = self.fl3(x3) if self.S[3] else self.v[3].repeat(x3.shape[0], 1)
        if self.nc == 2:
            x = torch.cat([fl0, fl1], dim=1)
        elif self.nc == 4:
            x = torch.cat([fl0, fl1, fl2, fl3], dim=1)
        # x = F.relu(self.fc3(x))
        x = self.fcf(x)
        x = F.softmax(x, dim=1)
        return x


class MyDenseNet121(nn.Module):
    def __init__(self, classCount=10, isTrained=True):
        super(MyDenseNet121, self).__init__()

        self.densenet121 = models.densenet121(pretrained=isTrained)
        kernelCount = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.densenet121(x)
        return x


class FLCNN(nn.Module):
    def __init__(self, classes=10, isTrained=False):
        super(FLCNN, self).__init__()
        """
        self.fl0 = FL0(cur_host=True) if cur_host == 0 else FL0(cur_host=False)
        self.fl1 = FL1(cur_host=True) if cur_host == 1 else FL1(cur_host=False)
        self.fl2 = FL2(cur_host=True) if cur_host == 2 else FL2(cur_host=False)
        self.fl3 = FL3(cur_host=True) if cur_host == 3 else FL3(cur_host=False)
        """

        nc = 2
        self.fl0 = MyDenseNet121(classCount=classes, isTrained=isTrained)
        self.fl1 = MyDenseNet121(classCount=classes, isTrained=isTrained)
        self.fcf = nn.Linear(nc*classes, classes)
        self.v = torch.zeros(nc, classes)  # fill-in
        self.S = torch.zeros(nc)  # set of clients

    def forward(self, x):

        if len(x) == 2:
            x0, x1 = x
        else:
            raise Exception('Invalid number of inputs.')

        fl0 = self.fl0(x0) if self.S[0] else self.v[0].repeat(x0.shape[0], 1)
        fl1 = self.fl1(x1) if self.S[1] else self.v[1].repeat(x1.shape[0], 1)

        x = torch.cat([fl0, fl1], dim=1)
        x = self.fcf(x)
        x = F.softmax(x, dim=1)
        return x