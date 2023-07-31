import os
import sys
import numpy as np
import PIL
import torch
import torch.utils.data as utils_data
from torchvision import datasets, transforms
import shutil
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

# set seed
np.random.seed(1226)
torch.manual_seed(1226)

normalize = transforms.Normalize(
    mean=[0.49088515, 0.48185424, 0.44636887],
    std=[0.20222517, 0.19923602, 0.20073999],
)


def check_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)


def download(url):
    import requests
    root = './data'
    check_folder(root)  # make sure data folder exists
    filename = root + '/' + url.split("/")[-1]
    exists = os.path.isfile(filename)
    if not exists:
        with open(filename, "wb") as f:
            r = requests.get(url)
            f.write(r.content)
    ftype = os.path.splitext(filename)[1]
    fname = filename[:-len(ftype)] + '.csv'
    exists = os.path.isfile(fname)
    if not exists:
        if ftype == '.gz':
            import gzip
            with gzip.open(filename, 'rb') as f_in:
                with open(fname, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        elif ftype == '.bz2':
            import bz2
            with bz2.open(filename, 'rb') as f_in:
                with open(fname, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
    return fname


def cropL(image: PIL.Image.Image) -> PIL.Image.Image:
    """Crop the images so only a specific region of interest is shown to my PyTorch model"""
    left, top, width, height = 0, 0, 24, 32
    return transforms.functional.crop(image, left=left, top=top, width=width, height=height)


def cropR(image: PIL.Image.Image) -> PIL.Image.Image:
    """Crop the images so only a specific region of interest is shown to my PyTorch model"""
    left, top, width, height = 8, 0, 24, 32
    return transforms.functional.crop(image, left=left, top=top, width=width, height=height)


def getData(data, num_workers=0, pin_memory=False):
    loader = utils_data.DataLoader(data, batch_size=len(data), num_workers=num_workers, pin_memory=pin_memory)
    data_iter = iter(loader)
    x, y = data_iter.next()
    return x, y


def cifar_loader(root='./data', batch_size=1, random_seed=1226, valid_size=0.2, shuffle=False, num_workers=0,
                 pin_memory=True, download=False):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - root: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    - test_loader: test set iteratior.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg
    check_folder(root)

    # define transforms
    transformL = transforms.Compose([
        transforms.Lambda(cropL),
        transforms.GaussianBlur(3),
        transforms.Pad([0, 0, 8, 0]),
        transforms.ToTensor(),
        normalize,
    ])
    transformR = transforms.Compose([
        transforms.Lambda(cropR),
        transforms.Pad([8, 0, 0, 0]),
        transforms.ToTensor(),
        normalize,
    ])

    # load the dataset
    dL = datasets.CIFAR10(root=root, train=True, download=download, transform=transformL)
    dR = datasets.CIFAR10(root=root, train=True, download=download, transform=transformR)
    xL, y = getData(dL)
    xR, _ = getData(dR)
    train_data = utils_data.TensorDataset(xL, xR, y)
    valid_data = utils_data.TensorDataset(xL, xR, y)
    dL = datasets.CIFAR10(root=root, train=False, download=download, transform=transformL)
    dR = datasets.CIFAR10(root=root, train=False, download=download, transform=transformR)
    xL, y = getData(dL)
    xR, _ = getData(dR)
    test_data = utils_data.TensorDataset(xL, xR, y)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = utils_data.SubsetRandomSampler(train_idx)
    valid_sampler = utils_data.SubsetRandomSampler(valid_idx)

    train_loader = utils_data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler,
                                         num_workers=num_workers, pin_memory=pin_memory)
    valid_loader = utils_data.DataLoader(valid_data, batch_size=batch_size, sampler=valid_sampler,
                                         num_workers=num_workers, pin_memory=pin_memory)
    test_loader = utils_data.DataLoader(test_data, batch_size=batch_size,
                                        num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def forest_loader(batch_size=1, seed=1226, state=1226, test_size=0.2, valid_size=0.2, num_workers=0, pin_memory=True,
                  u='https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz', std=1,
                  c0=[*range(0, 6), 9, *range(14, 54)], c1=[*range(6, 9), *range(10, 54)], adv=[], adv_valid=True):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load Data
    filename2 = download(u)

    # import dataset
    data = pd.read_csv(filename2, header=None)

    X = data.values[:, :-1]
    X = X.reshape((X.shape[0], 54))
    y = data.values[:, -1] - 1

    X, X_test, y, y_test = train_test_split(np.array(X), np.array(y), test_size=test_size, random_state=state)
    X, X_valid, y, y_valid = train_test_split(np.array(X), np.array(y), test_size=valid_size, random_state=state)

    # normalize data
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    # convert data-types
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).long()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).long()
    X_valid = torch.from_numpy(X_valid).float()
    y_valid = torch.from_numpy(y_valid).long()

    x1, x2 = X[:, c0], X[:, c1]
    if len(adv) > 0:
        x1[:, adv] += torch.normal(mean=0, std=std, size=(x1.shape[0], len(adv)))
    train_data = utils_data.TensorDataset(x1, x2, y)
    x1, x2 = X_valid[:, c0], X_valid[:, c1]
    if len(adv) > 0 and adv_valid:
        x1[:, adv] += torch.normal(mean=0, std=std, size=(x1.shape[0], len(adv)))
    valid_data = utils_data.TensorDataset(x1, x2, y_valid)
    x1, x2 = X_test[:, c0], X_test[:, c1]
    if len(adv) > 0 and adv_valid:
        x1[:, adv] += torch.normal(mean=0, std=std, size=(x1.shape[0], len(adv)))
    test_data = utils_data.TensorDataset(x1, x2, y_test)

    train_loader = utils_data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers,
                                         pin_memory=pin_memory)
    valid_loader = utils_data.DataLoader(valid_data, batch_size=batch_size, num_workers=num_workers,
                                         pin_memory=pin_memory)
    test_loader = utils_data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers,
                                        pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def adv_forest_loader(batch_size=1, num_workers=0, pin_memory=True, split=14, head='advLogReg', adv_valid=True,
                      u='https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz',
                      c0=[*range(0, 6), 9, *range(14, 54)], c1=[*range(6, 9), *range(10, 54)], random_seed=1226,
                      test_size=0.2, valid_size=0.2):

    check_folder('./data/')

    # import data
    X = np.genfromtxt('./data/' + head + '.csv', delimiter=',')
    y = np.genfromtxt('./data/' + head + '_y.csv', delimiter=',')
    if adv_valid:
        X_valid = np.genfromtxt('./data/' + head + '_valid.csv', delimiter=',')
        y_valid = np.genfromtxt('./data/' + head + '_y_valid.csv', delimiter=',')
        X_test = np.genfromtxt('./data/' + head + '_test.csv', delimiter=',')
        y_test = np.genfromtxt('./data/' + head + '_y_test.csv', delimiter=',')
    else:
        # Load Data
        filename2 = download(u)

        # import dataset
        data = pd.read_csv(filename2, header=None)

        Xg = data.values[:, :-1]
        Xg = Xg.reshape((Xg.shape[0], 54))
        yg = data.values[:, -1] - 1

        Xg, X_test, yg, y_test = train_test_split(np.array(Xg), np.array(yg), test_size=test_size,
                                                  random_state=random_seed)
        Xg, X_valid, yg, y_valid = train_test_split(np.array(Xg), np.array(yg), test_size=valid_size,
                                                    random_state=random_seed)

        # normalize data
        scaler = StandardScaler()
        scaler.fit(Xg)
        X_valid = scaler.transform(X_valid)
        X_test = scaler.transform(X_test)

    # convert data-types
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).long()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).long()
    X_valid = torch.from_numpy(X_valid).float()
    y_valid = torch.from_numpy(y_valid).long()

    x1, x2 = X[:, :split], X[:, split:]
    train_data = utils_data.TensorDataset(x1, x2, y)
    if adv_valid:
        x1, x2 = X_valid[:, :split], X_valid[:, split:]
    else:
        x1, x2 = X_valid[:, c0], X_valid[:, c1]
    valid_data = utils_data.TensorDataset(x1, x2, y_valid)
    if adv_valid:
        x1, x2 = X_test[:, :split], X_test[:, split:]
    else:
        x1, x2 = X_test[:, c0], X_test[:, c1]
    test_data = utils_data.TensorDataset(x1, x2, y_test)

    train_loader = utils_data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers,
                                         pin_memory=pin_memory)
    valid_loader = utils_data.DataLoader(valid_data, batch_size=batch_size, num_workers=num_workers,
                                         pin_memory=pin_memory)
    test_loader = utils_data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers,
                                        pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def taiwan_loader(batch_size=1, test_size=0.2, seed=1226, state=1226, valid_size=0.2, num_workers=0, pin_memory=True,
                  c0=[], c1=[], adv=[], adv_valid=True, u='taiwan.csv'):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # import dataset
    data = pd.read_csv('./data/' + u, header=0)

    X = data.values[:, 1:]
    y = data.values[:, 0]

    X, X_test, y, y_test = train_test_split(np.array(X), np.array(y), test_size=test_size, random_state=state)
    X, X_valid, y, y_valid = train_test_split(np.array(X), np.array(y), test_size=valid_size, random_state=state)

    # normalize data
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    # convert data-types
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).long()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).long()
    X_valid = torch.from_numpy(X_valid).float()
    y_valid = torch.from_numpy(y_valid).long()

    x1, x2 = X[:, c0], X[:, c1]
    if len(adv) > 0:
        x1[:, adv] += torch.normal(mean=0, std=std, size=(x1.shape[0], len(adv)))
    train_data = utils_data.TensorDataset(x1, x2, y)
    x1, x2 = X_valid[:, c0], X_valid[:, c1]
    if len(adv) > 0 and adv_valid:
        x1[:, adv] += torch.normal(mean=0, std=std, size=(x1.shape[0], len(adv)))
    valid_data = utils_data.TensorDataset(x1, x2, y_valid)
    x1, x2 = X_test[:, c0], X_test[:, c1]
    if len(adv) > 0 and adv_valid:
        x1[:, adv] += torch.normal(mean=0, std=std, size=(x1.shape[0], len(adv)))
    test_data = utils_data.TensorDataset(x1, x2, y_test)

    train_loader = utils_data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers,
                                         pin_memory=pin_memory)
    valid_loader = utils_data.DataLoader(valid_data, batch_size=batch_size, num_workers=num_workers,
                                         pin_memory=pin_memory)
    test_loader = utils_data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers,
                                        pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def replace_str_cols(co, col_names):
    cols = []
    for cn in co:
        cols.extend([c for c in range(len(col_names)) if col_names[c].startswith(cn)])
    return cols


with open('ni_cols_orig.txt', 'r') as f:
    cols_orig = f.read().split('\n')


def ni_loader(batch_size=1, seed=1226, state=1226, valid_size=0.2, num_workers=0, pin_memory=True, std=1,
                  c0=[], c1=[], adv=[], adv_valid=True, classes=2):
    np.random.seed(seed)
    torch.manual_seed(seed)

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

    X = inputs.values[:train_len, :-classes]
    y_onehot = inputs.values[:train_len, -classes:]
    y = np.asarray([np.where(r == 1)[0][0] for r in y_onehot])  # convert from one-hot to integer encoding

    X_test = inputs.values[train_len:, :-classes]
    y_test_onehot = inputs.values[train_len:, -classes:]
    y_test = np.asarray([np.where(r == 1)[0][0] for r in y_test_onehot])  # convert from one-hot to integer encoding

    X = X.reshape((train_len, train_feat))
    X_test = X_test.reshape((test.shape[0], train_feat))

    X, X_valid, y, y_valid = train_test_split(np.array(X), np.array(y), test_size=valid_size, random_state=state)

    # normalize data
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    # convert data-types
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).long()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).long()
    X_valid = torch.from_numpy(X_valid).float()
    y_valid = torch.from_numpy(y_valid).long()

    x1, x2 = X[:, c0], X[:, c1]
    if len(adv) > 0:
        x1[:, adv] += torch.normal(mean=0, std=std, size=(x1.shape[0], len(adv)))
    train_data = utils_data.TensorDataset(x1, x2, y)
    x1, x2 = X_valid[:, c0], X_valid[:, c1]
    if len(adv) > 0 and adv_valid:
        x1[:, adv] += torch.normal(mean=0, std=std, size=(x1.shape[0], len(adv)))
    valid_data = utils_data.TensorDataset(x1, x2, y_valid)
    x1, x2 = X_test[:, c0], X_test[:, c1]
    if len(adv) > 0 and adv_valid:
        x1[:, adv] += torch.normal(mean=0, std=std, size=(x1.shape[0], len(adv)))
    test_data = utils_data.TensorDataset(x1, x2, y_test)

    train_loader = utils_data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers,
                                         pin_memory=pin_memory)
    valid_loader = utils_data.DataLoader(valid_data, batch_size=batch_size, num_workers=num_workers,
                                         pin_memory=pin_memory)
    test_loader = utils_data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers,
                                        pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader
