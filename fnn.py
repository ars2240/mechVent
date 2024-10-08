import itertools
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
    def __init__(self, train_feat, nc=4, classes=5, h1=None, h2=None, doub=False, seed=1226):
        super(FLN, self).__init__()
        torch.manual_seed(seed)
        """
        self.fl0 = FL0(cur_host=True) if cur_host == 0 else FL0(cur_host=False)
        self.fl1 = FL1(cur_host=True) if cur_host == 1 else FL1(cur_host=False)
        self.fl2 = FL2(cur_host=True) if cur_host == 2 else FL2(cur_host=False)
        self.fl3 = FL3(cur_host=True) if cur_host == 3 else FL3(cur_host=False)
        """
        self.nc = nc
        self.classes = classes
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
        self.v = torch.zeros(nc, classes).requires_grad_()  # fill-in
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
        if self.classes == 1:
            x = torch.sigmoid(x)
            x = torch.squeeze(x, dim=1)
        else:
            x = F.softmax(x, dim=1)
        return x


class FLR(nn.Module):
    def __init__(self, train_feat, nc=4, classes=5, seed=1226):
        torch.manual_seed(seed)
        super(FLR, self).__init__()
        self.nc = nc
        self.classes = classes
        self.train_feat = train_feat
        if nc == 2:
            self.fl0, self.fl1 = nn.Linear(train_feat[0], classes), nn.Linear(train_feat[1], classes)
        elif nc == 4:
            self.fl0, self.fl1, self.fl2, self.fl3 = nn.Linear(train_feat[0], classes),\
                                                     nn.Linear(train_feat[1], classes),\
                                                     nn.Linear(train_feat[2], classes),\
                                                     nn.Linear(train_feat[3], classes)
        else:
            raise Exception('Invalid number of inputs.')
        self.v = torch.zeros(nc, classes).requires_grad_()  # fill-in
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

        x0 = x0.reshape(x0.shape[0], -1)
        fl0 = self.fl0(x0) if self.S[0] else self.v[0].repeat(x0.shape[0], 1)
        x1 = x1.reshape(x1.shape[0], -1)
        fl1 = self.fl1(x1) if self.S[1] else self.v[1].repeat(x1.shape[0], 1)
        if self.nc >= 3:
            x2 = x2.reshape(x2.shape[0], -1)
            fl2 = self.fl2(x2) if self.S[2] else self.v[2].repeat(x2.shape[0], 1)
        if self.nc >= 4:
            x3 = x3.reshape(x3.shape[0], -1)
            fl3 = self.fl3(x3) if self.S[3] else self.v[3].repeat(x3.shape[0], 1)

        if self.nc == 2:
            x = fl0 + fl1
        elif self.nc == 4:
            x = fl0 + fl1 + fl2 + fl3

        if self.classes == 1:
            x = torch.sigmoid(x)
            x = torch.squeeze(x, dim=1)
        else:
            x = F.softmax(x, dim=1)
        return x


class FLRSH(nn.Module):
    def __init__(self, feats, nc=4, classes=5, seed=1226):
        torch.manual_seed(seed)
        super(FLRSH, self).__init__()

        self.nc = nc
        self.classes = classes
        self.train_feat = [len(f) for f in feats]
        self.c, self.cl, loc = [None] * nc, [None] * nc, [None] * nc
        self.shared = [f for f in feats[0] if f in feats[1]]
        self.shl = len(self.shared)
        self.sh = nn.Linear(self.shl, classes)
        for i in range(nc):
            self.c[i] = [f for f in feats[i] if f not in self.shared]
            self.cl[i] = len(self.c[i])
            loc[i] = nn.Linear(self.cl[i], classes)
        self.loc = nn.ModuleList(loc)
        self.v = torch.zeros(nc, classes).requires_grad_()  # fill-in
        self.S = torch.zeros(nc)  # set of clients

    def forward(self, x):

        if len(x) != self.nc:
            raise Exception('Invalid number of inputs.')

        fl = [None] * self.nc
        for i in range(self.nc):
            x2 = x[i]
            s, f = x2.shape[0], self.cl[i]
            x2 = x2.reshape(s, -1)
            fl[i] = self.loc[i](x2[:, :f]) + self.sh(x2[:, f:]) if self.S[i] else self.v[i].repeat(s, 1)

        x = sum(fl)

        if self.classes == 1:
            x = torch.sigmoid(x)
            x = torch.squeeze(x, dim=1)
        else:
            x = F.softmax(x, dim=1)
        return x


class FLRHZ(nn.Module):
    def __init__(self, feats, nf=[10, 10], m=1, nc=4, classes=5, seed=1226):
        torch.manual_seed(seed)
        super(FLRHZ, self).__init__()

        self.nc = nc
        self.classes = classes
        self.train_feat = [len(f) for f in feats]
        self.c, self.cl, loc = [None] * nc, [None] * nc, [None] * nc
        self.shared = [f for f in feats[0] if f in feats[1]]
        self.shl = len(self.shared)
        self.embed_sh = nn.ModuleList([nn.Linear(self.shl * m, nf[0]) for _ in range(nc)])
        for i in range(nc):
            self.c[i] = [f for f in feats[i] if f not in self.shared]
            self.cl[i] = len(self.c[i]) * m
            loc[i] = nn.Linear(self.cl[i], nf[1])
        self.embed_loc = nn.ModuleList(loc)
        self.f = nn.ModuleList([nn.Linear(sum(nf), classes) for _ in range(nc)])
        self.S = torch.zeros(nc)  # set of clients

    def forward(self, x, client=None):

        if len(x) == 1:
            x = x[0]
        if len(x) != self.nc:
            raise Exception('Invalid number of inputs: {0} != {1}.'.format(len(x), self.nc))

        if client is None:

            fl = [None] * sum(self.S)
            idx = [i for i in range(self.nc) if self.S[i]]
            ni = len(idx)
            for i in range(ni):
                c = idx[i]
                x2 = x[c]
                s, f = x2.shape[0], self.cl[c]
                x2 = x2.reshape(s, -1)
                fl[i] = self.f[c](torch.cat((self.embed_loc[c](x2[:, :f]), self.embed_sh[c](x2[:, f:])), dim=1))

            x = sum(fl)/ni
        else:
            x2 = x[client]
            s, f = x2.shape[0], self.cl[client]
            x2 = x2.reshape(s, -1)
            x = self.f[client](torch.cat((self.embed_loc[client](x2[:, :f]), self.embed_sh[client](x2[:, f:])), dim=1))

        if self.classes == 1:
            x = torch.sigmoid(x)
            x = torch.squeeze(x, dim=1)
        else:
            x = F.softmax(x, dim=1)
        return x


class FCNNHZ(nn.Module):
    def __init__(self, feats, nc=4, classes=5, seed=1226):
        torch.manual_seed(seed)
        super(FCNNHZ, self).__init__()

        self.nc = nc
        self.classes = classes
        self.train_feat = [len(f) for f in feats]
        resnet = [None] * nc
        for i in range(nc):
            resnet[i] = models.resnet18()
            resnet[i].fc = nn.Linear(resnet[i].fc.in_features, classes)
        self.f = nn.ModuleList(resnet)
        self.S = torch.zeros(nc)  # set of clients

    def good_img_mat(self, good_imgs):
        nimgs = torch.sum(good_imgs, dim=1)
        l = len(nimgs)
        mat = torch.zeros(l, torch.sum(nimgs).item())
        tot = 0
        for i in range(l):
            ni = nimgs[i]
            mat[i, tot:(tot + ni)] = 1/ni
            tot += ni
        return mat

    def net(self, x, c):
        x2 = x[c]
        s = x2.shape
        # print('s[{0}]: {1}'.format(c, s))
        good_imgs = ~torch.any(x2.isnan(), dim=(2, 3, 4))
        x2 = x2.reshape(s[0] * s[1], s[2], s[3], s[4])
        x2 = x2[good_imgs.reshape(-1)]
        x2 = self.f[c](x2)
        x2 = torch.matmul(self.good_img_mat(good_imgs), x2)
        return x2

    def forward(self, x, client=None):

        if len(x) == 1:
            x = x[0]
        if len(x) != self.nc:
            raise Exception('Invalid number of inputs: {0} != {1}.'.format(len(x), self.nc))

        if client is None:

            fl = [None] * sum(self.S)
            idx = [i for i in range(self.nc) if self.S[i]]
            ni = len(idx)
            for i in range(ni):
                c = idx[i]
                fl[i] = self.net(x, c)

            x = sum(fl)/ni
        else:
            x = self.net(x, client)

        # print('Post-ResNet shape: {0}'.format(x.shape))

        if self.classes == 1:
            x = torch.sigmoid(x)
            x = torch.squeeze(x, dim=1)
        else:
            x = F.softmax(x, dim=1)
        # print('Final shape: {0}'.format(x.shape))
        return x


class FLNSH(nn.Module):
    def __init__(self, feats, nc=4, hidden=[5, 5], classes=5, seed=1226):
        torch.manual_seed(seed)
        super(FLNSH, self).__init__()
        self.nc = nc
        self.classes = classes
        self.train_feat = [len(f) for f in feats]
        self.c, self.cl, loc0, loc1 = [None] * nc, [None] * nc, [None] * nc, [None] * nc
        self.shared = [f for f in feats[0] if f in feats[1]]
        self.shl = len(self.shared)
        self.sh0, self.sh1 = (nn.Linear(self.shl, hidden[1]), nn.Linear(hidden[1], classes))
        for i in range(nc):
            self.c[i] = [f for f in feats[i] if f not in self.shared]
            self.cl[i] = len(self.c[i])
            loc0[i], loc1[i] = nn.Linear(self.cl[i], hidden[0]), nn.Linear(hidden[0], classes)
        self.loc0, self.loc1 = nn.ModuleList(loc0), nn.ModuleList(loc1)
        self.v = torch.zeros(nc, classes).requires_grad_()  # fill-in
        self.S = torch.zeros(nc)  # set of clients

    def forward(self, x):

        if len(x) != self.nc:
            raise Exception('Invalid number of inputs.')

        fl = [None] * self.nc
        for i in range(self.nc):
            x2 = x[i]
            s, f = x2.shape[0], self.cl[i]
            x2 = x2.reshape(s, -1)
            loc, sh = F.relu(self.loc0[i](x2[:, :f])), F.relu(self.sh0(x2[:, f:]))
            fl[i] = self.loc1[i](loc) + self.sh1(sh) if self.S[i] else self.v[i].repeat(s, 1)

        x = sum(fl)

        if self.classes == 1:
            x = torch.sigmoid(x)
            x = torch.squeeze(x, dim=1)
        else:
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
        self.classes = classes
        self.fl0 = MyDenseNet121(classCount=classes, isTrained=isTrained)
        self.fl1 = MyDenseNet121(classCount=classes, isTrained=isTrained)
        self.fcf = nn.Linear(nc*classes, classes)
        self.v = torch.zeros(nc, classes).requires_grad_()  # fill-in
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
        if self.classes == 1:
            x = torch.sigmoid(x)
            x = torch.squeeze(x, dim=1)
        else:
            x = F.softmax(x, dim=1)
        return x