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

# set seed
np.random.seed(1226)
torch.manual_seed(1226)


with open('ni_cols_orig.txt', 'r') as f:
    cols_orig = f.read().split('\n')

one_hot = ['protocol_type', 'service', 'flag']


def check_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)


def to_loader(inputs, target, use_gpu=False, batch_size=128, num_workers=0):
    if isinstance(inputs, list):
        i0, i1, i2, i3 = inputs
        data = utils_data.TensorDataset(i0, i1, i2, i3, target)
    else:
        data = utils_data.TensorDataset(inputs, target)
    if use_gpu:
        loader = utils_data.DataLoader(data, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    else:
        loader = utils_data.DataLoader(data, batch_size=batch_size)
    return loader


def shuffle_along_axis(a, axis=0):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)


def shuffle(X, cols, swap, axis=1):
    n = round(X.shape[0] * swap)
    rows = np.random.choice(X.shape[0], n, replace=False)

    # print(X[rows[0], cols])
    X[np.ix_(rows, cols)] = shuffle_along_axis(X[np.ix_(rows, cols)], axis=axis)
    # print(X[rows[0], cols])
    # print(np.sum(X[np.ix_(rows, cols)], axis=1))

    return X


def ni_loader(batch_size=128, use_gpu=False, shared='service', cur_host=0, swap=0.0, cor_test=False, classes=5,
              reduced=False, double=False):
    if isinstance(shared, str):
        shared = [shared]

    if reduced:
        col_dic = {
            0: ['src_bytes', 'service', 'flag'],
            1: ['logged_in', 'num_shells'],
            2: [],
            3: ['dst_host_same_srv_rate', 'dst_host_srv_diff_host_rate']}
    else:
        col_dic = {
            0: ['duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'protocol_type', 'service',
                'flag'],
            1: ['hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
                'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
                'is_guest_login'],
            2: ['count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
                'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate'],
            3: ['dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']}

    # Load Data
    u = 'http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz'
    filename2 = download(u)
    u = 'http://kdd.ics.uci.edu/databases/kddcup99/corrected.gz'
    filename3 = download(u)

    # import dataset
    train = pd.read_csv(filename2, header=None)
    test = pd.read_csv(filename3, header=None)

    # transform to supercategories
    if classes == 2:
        dic = {'normal.': 'normal', 'land.': 'dos', 'pod.': 'dos', 'teardrop.': 'dos', 'back.': 'dos',
               'neptune.': 'dos', 'smurf.': 'dos'}
    elif classes == 5:
        dic = {'normal.': 'normal', 'nmap.': 'probing', 'portsweep.': 'probing', 'ipsweep.': 'probing',
               'satan.': 'probing', 'land.': 'dos', 'pod.': 'dos', 'teardrop.': 'dos', 'back.': 'dos',
               'neptune.': 'dos', 'smurf.': 'dos', 'spy.': 'r2l', 'phf.': 'r2l', 'multihop.': 'r2l',
               'ftp_write.': 'r2l', 'imap.': 'r2l', 'warezmaster.': 'r2l', 'guess_passwd.': 'r2l',
               'buffer_overflow.': 'u2r', 'rootkit.': 'u2r', 'loadmodule.': 'u2r', 'perl.': 'u2r'}
    else:
        raise Exception('Invalid number of classes')
    i = train.shape[1] - 1
    train = train.loc[train[i].isin(dic.keys())]
    train.replace({i: dic}, inplace=True)
    test = test.loc[test[i].isin(dic.keys())]
    test.replace({i: dic}, inplace=True)

    train_len = train.shape[0]  # save length of training set
    train = pd.concat([train, test], ignore_index=True)
    train.columns = cols_orig
    inputs = pd.get_dummies(train)  # convert objects to one-hot encoding
    train_feat = inputs.shape[1] - classes  # number of features
    col_names = list(inputs.columns)
    # print(col_names)
    """
    with open('col_names.txt', 'w') as f:
        for item in col_names:
            # write each item on a new line
            f.write("%s\n" % item)
    """

    X = inputs.values[:train_len, :-classes]
    y_onehot = inputs.values[:train_len, -classes:]
    y = np.asarray([np.where(r == 1)[0][0] for r in y_onehot])  # convert from one-hot to integer encoding

    X_test = inputs.values[train_len:, :-classes]
    y_test_onehot = inputs.values[train_len:, -classes:]
    y_test = np.asarray([np.where(r == 1)[0][0] for r in y_test_onehot])  # convert from one-hot to integer encoding

    X = X.reshape((train_len, train_feat))
    X_test = X_test.reshape((test.shape[0], train_feat))

    # X_old, X_told = np.copy(X), np.copy(X_test)
    # shuffle data
    if swap > 0:
        for cn in shared:
            cols = [c for c in range(len(col_names)) if col_names[c].startswith(cn)]
            if cn in one_hot:
                X = shuffle(X, cols, swap)
                if cor_test:
                    X_test = shuffle(X_test, cols, swap)
            else:
                X = shuffle(X, cols, swap, axis=0)
                if cor_test:
                    X_test = shuffle(X_test, cols, swap, axis=0)
    # print(np.linalg.norm(X - X_old))
    # print(np.linalg.norm(X_test - X_told))
    # raise Exception

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

    X1, X2 = [], []
    train_feat = []
    for i in col_dic.keys():
        cd = [c for c in col_dic[i] if any(not c.startswith(cn) for cn in shared)]
        cols = [c for c in range(len(col_names)) if any(col_names[c].startswith(cn) for cn in cd)]
        if i == cur_host:
            sc = [c for c in range(len(col_names)) if any(col_names[c].startswith(cn) for cn in shared)]
            cols.extend(sc)
        X1.append(X[:, cols])
        X2.append(X_test[:, cols])
        train_feat.append(len(cols))

    model = FLN(train_feat, classes, double)

    train_data = to_loader(X1, y, use_gpu, batch_size)
    test_data = to_loader(X2, y_test, use_gpu, batch_size)

    return train_data, test_data, model


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


# Federated Learning Architecture
class FL0(nn.Module):
    def __init__(self, cur_host=False):
        super(FL0, self).__init__()
        train_feat = 20
        train_feat += 71 if cur_host else 0
        # h1 = 10 if cur_host else 5
        # out = 6 if cur_host else 3
        h1, out = 5, 3
        self.fc1 = nn.Linear(train_feat, h1)
        self.fc2 = nn.Linear(h1, out)

    def forward(self, x):
        if type(x).__module__ == 'numpy':
            x = torch.from_numpy(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class FL1(nn.Module):
    def __init__(self, cur_host=False):
        super(FL1, self).__init__()
        train_feat = 13
        train_feat += 71 if cur_host else 0
        # h1 = 10 if cur_host else 5
        # out = 6 if cur_host else 3
        h1, out = 5, 3
        self.fc1 = nn.Linear(train_feat, h1)
        self.fc2 = nn.Linear(h1, out)

    def forward(self, x):
        if type(x).__module__ == 'numpy':
            x = torch.from_numpy(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class FL2(nn.Module):
    def __init__(self, cur_host=False):
        super(FL2, self).__init__()
        train_feat = 9
        train_feat += 71 if cur_host else 0
        # h1 = 10 if cur_host else 5
        # out = 6 if cur_host else 3
        h1, out = 5, 3
        self.fc1 = nn.Linear(train_feat, h1)
        self.fc2 = nn.Linear(h1, out)

    def forward(self, x):
        if type(x).__module__ == 'numpy':
            x = torch.from_numpy(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class FL3(nn.Module):
    def __init__(self, cur_host=False):
        super(FL3, self).__init__()
        train_feat = 10
        train_feat += 71 if cur_host else 0
        # h1 = 10 if cur_host else 5
        # out = 6 if cur_host else 3
        h1, out = 5, 3
        self.fc1 = nn.Linear(train_feat, h1)
        self.fc2 = nn.Linear(h1, out)

    def forward(self, x):
        if type(x).__module__ == 'numpy':
            x = torch.from_numpy(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class FLC(nn.Module):
    def __init__(self, train_feat, double=False):
        super(FLC, self).__init__()
        h1, out = 1, 1
        self.fc1 = nn.Linear(train_feat, h1)
        self.double = double
        # self.fc2 = nn.Linear(h1, out)

    def forward(self, x):
        if type(x).__module__ == 'numpy':
            x = torch.from_numpy(x)
        if self.double:
            x = x.double()

        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        return x


class FLN(nn.Module):
    def __init__(self, train_feat, classes=5, double=False):
        super(FLN, self).__init__()
        """
        self.fl0 = FL0(cur_host=True) if cur_host == 0 else FL0(cur_host=False)
        self.fl1 = FL1(cur_host=True) if cur_host == 1 else FL1(cur_host=False)
        self.fl2 = FL2(cur_host=True) if cur_host == 2 else FL2(cur_host=False)
        self.fl3 = FL3(cur_host=True) if cur_host == 3 else FL3(cur_host=False)
        """
        self.train_feat = train_feat
        self.fl0, self.fl1, self.fl2, self.fl3 = FLC(train_feat[0], double), FLC(train_feat[1], double),\
                                                 FLC(train_feat[2], double), FLC(train_feat[3], double)
        # self.fc3 = nn.Linear(4, 5)
        self.fc4 = nn.Linear(4, classes)

    def forward(self, x):

        if len(x) == 4:
            x0, x1, x2, x3 = x
        else:
            tf = np.cumsum(self.train_feat)
            x0, x1, x2, x3 = x[:, :tf[0]], x[:, tf[0]:tf[1]], x[:, tf[1]:tf[2]], x[:, tf[2]:]
        x = torch.cat([self.fl0(x0), self.fl1(x1), self.fl2(x2), self.fl3(x3)], dim=1)
        # x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = F.softmax(x, dim=1)
        return x

