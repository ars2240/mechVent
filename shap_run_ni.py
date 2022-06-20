from shap_model import *
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils_data
from sklearn.preprocessing import StandardScaler
import pandas as pd

h = '/Users/adamsandler/Documents/Northwestern/Research/Optimization with Bounded Eigenvalues/'
sys.path.insert(1, h)

from opt import download

use_gpu = False

# set seed
np.random.seed(1226)
torch.manual_seed(1226)

# Parameters
batch_size = 128

# Load Data
u = 'http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz'
filename2 = download(u)
u = 'http://kdd.ics.uci.edu/databases/kddcup99/corrected.gz'
filename3 = download(u)

# import dataset
train = pd.read_csv(filename2, header=None)
test = pd.read_csv(filename3, header=None)

# transform to supercategories
dic = {'normal.': 'normal',  'nmap.': 'probing', 'portsweep.': 'probing', 'ipsweep.': 'probing', 'satan.': 'probing',
       'land.': 'dos', 'pod.': 'dos', 'teardrop.': 'dos', 'back.': 'dos', 'neptune.': 'dos', 'smurf.': 'dos',
       'spy.': 'r2l', 'phf.': 'r2l', 'multihop.': 'r2l', 'ftp_write.': 'r2l', 'imap.': 'r2l', 'warezmaster.': 'r2l',
       'guess_passwd.': 'r2l', 'buffer_overflow.': 'u2r', 'rootkit.': 'u2r', 'loadmodule.': 'u2r', 'perl.': 'u2r'}
i = train.shape[1] - 1
train = train.loc[train[i].isin(dic.keys())]
train.replace({i: dic}, inplace=True)
test = test.loc[test[i].isin(dic.keys())]
test.replace({i: dic}, inplace=True)

train_len = train.shape[0]  # save length of training set
train = train.append(test, ignore_index=True)
inputs = pd.get_dummies(train)  # convert objects to one-hot encoding
train_feat = inputs.shape[1] - 5  # number of features

X = inputs.values[:train_len, :-5]
y_onehot = inputs.values[:train_len, -5:]
y = np.asarray([np.where(r == 1)[0][0] for r in y_onehot])  # convert from one-hot to integer encoding

# class balance
cts = []
for i in set(y):
    cts.append(list(y).count(i))

X_test = inputs.values[train_len:, :-5]
y_test_onehot = inputs.values[train_len:, -5:]
y_test = np.asarray([np.where(r == 1)[0][0] for r in y_test_onehot])  # convert from one-hot to integer encoding

# class balance
cts = []
for i in set(y_test):
    cts.append(list(y_test).count(i))

X = X.reshape((train_len, train_feat))
X_test = X_test.reshape((test.shape[0], train_feat))

# normalize data
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
X_test = scaler.transform(X_test)

# convert data-types
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).long()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).long()


# Neural Network Architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(train_feat, 13)
        self.fc2 = nn.Linear(13, 15)
        self.fc3 = nn.Linear(15, 20)
        self.fc4 = nn.Linear(20, 5)

    def forward(self, x):
        if type(x).__module__ == 'numpy':
            x = torch.from_numpy(x)
        x = x.double()

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = F.softmax(x, dim=1)
        return x


def to_loader(inputs, target, use_gpu=False, batch_size=128, num_workers=0):
    data = utils_data.TensorDataset(inputs, target)
    if use_gpu:
        loader = utils_data.DataLoader(data, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    else:
        loader = utils_data.DataLoader(data, batch_size=batch_size)
    return loader


train_data = to_loader(X, y, use_gpu, batch_size)
test_data = to_loader(X_test, y_test, use_gpu, batch_size)
model = Net()


def alpha(k):
    return 1e-4 / (1 + k)


model_id = "NI_SGD_mu0_K0_trained_model_best"
s = shap(model, h + 'models/' + model_id + '.pt', alpha=alpha, use_gpu=use_gpu)
s.explainer(train_data, test_data)
# s.run(train_loader, test_loader)
