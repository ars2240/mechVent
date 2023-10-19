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


def getData(data, num_workers=0, pin_memory=False):
    loader = utils_data.DataLoader(data, batch_size=len(data), num_workers=num_workers, pin_memory=pin_memory)
    data_iter = iter(loader)
    x, y = data_iter.next()
    return x, y


def getLoaders(X, X_valid, X_test, y, y_valid, y_test, batch_size=1, seed=1226, num_workers=0, pin_memory=True, std=1,
               c=[], adv=[], adv_valid=True, counts=False, scale=True):
    torch.manual_seed(seed)

    # normalize data
    if scale:
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        X_valid = scaler.transform(X_valid)
        X_test = scaler.transform(X_test)

    if counts:
        print('Train counts: {0}'.format({item: list(y).count(item)/len(y) for item in y}))
        print('Valid counts: {0}'.format({item: list(y_valid).count(item)/len(y_valid) for item in y_valid}))
        print('Test counts: {0}'.format({item: list(y_test).count(item)/len(y_test) for item in y_test}))

    # convert data-types
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).long()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).long()
    X_valid = torch.from_numpy(X_valid).float()
    y_valid = torch.from_numpy(y_valid).long()

    nc = len(c)
    x = [None] * nc
    for i in range(nc):
        x[i] = X[:, c[i]]
    if isinstance(adv, list) and len(adv) > 0:
        x[0][:, adv] += torch.normal(mean=0, std=std, size=(x[0].shape[0], len(adv)))
    elif isinstance(adv, dict):
        for a in adv.keys():
            x[a][:, adv[a]] += torch.normal(mean=0, std=std, size=(x[a].shape[0], len(adv[a])))
    train_data = utils_data.TensorDataset(*x, y)
    for i in range(nc):
        x[i] = X_valid[:, c[i]]
    if isinstance(adv, list) and len(adv) > 0 and adv_valid:
        x[0][:, adv] += torch.normal(mean=0, std=std, size=(x[0].shape[0], len(adv)))
    elif isinstance(adv, dict) and adv_valid:
        for a in adv.keys():
            x[a][:, adv[a]] += torch.normal(mean=0, std=std, size=(x[a].shape[0], len(adv[a])))
    valid_data = utils_data.TensorDataset(*x, y_valid)
    for i in range(nc):
        x[i] = X_test[:, c[i]]
    if isinstance(adv, list) and len(adv) > 0 and adv_valid:
        x[0][:, adv] += torch.normal(mean=0, std=std, size=(x[0].shape[0], len(adv)))
    elif isinstance(adv, dict) and adv_valid:
        for a in adv.keys():
            x[a][:, adv[a]] += torch.normal(mean=0, std=std, size=(x[a].shape[0], len(adv[a])))
    test_data = utils_data.TensorDataset(*x, y_test)

    train_loader = utils_data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers,
                                         pin_memory=pin_memory)
    valid_loader = utils_data.DataLoader(valid_data, batch_size=batch_size, num_workers=num_workers,
                                         pin_memory=pin_memory)
    test_loader = utils_data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers,
                                        pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def cifar_loader(root='./data', batch_size=1, seed=1226, valid_size=0.2, shuffle=False, num_workers=0,
                 pin_memory=True, download=False, crop=8, blur=3, pad=True):
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

    np.random.seed(seed)
    torch.manual_seed(seed)

    def cropL(image: PIL.Image.Image) -> PIL.Image.Image:
        """Crop the images so only a specific region of interest is shown to my PyTorch model"""
        left, top, width, height = 0, 0, 32 - crop, 32
        return transforms.functional.crop(image, left=left, top=top, width=width, height=height)

    def cropR(image: PIL.Image.Image) -> PIL.Image.Image:
        """Crop the images so only a specific region of interest is shown to my PyTorch model"""
        left, top, width, height = crop, 0, 32 - crop, 32
        return transforms.functional.crop(image, left=left, top=top, width=width, height=height)

    # define transforms
    if pad:
        transformL = transforms.Compose([
            transforms.Lambda(cropL),
            transforms.GaussianBlur(blur),
            transforms.Pad([0, 0, crop, 0]),
            transforms.ToTensor(),
            normalize,
        ])
        transformR = transforms.Compose([
            transforms.Lambda(cropR),
            transforms.Pad([crop, 0, 0, 0]),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transformL = transforms.Compose([
            transforms.Lambda(cropL),
            transforms.GaussianBlur(blur),
            transforms.ToTensor(),
            normalize,
        ])
        transformR = transforms.Compose([
            transforms.Lambda(cropR),
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
                  c=[], adv=[], adv_valid=True, counts=False):
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

    train_loader, valid_loader, test_loader = getLoaders(X, X_valid, X_test, y, y_valid, y_test, batch_size, seed,
                                                         num_workers, pin_memory, std, c, adv, adv_valid, counts, True)

    return train_loader, valid_loader, test_loader


def adv_forest_loader(batch_size=1, num_workers=0, pin_memory=True, split=None, head='advLogReg', adv_valid=True,
                      u='https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz',
                      c0=[], c1=[], random_seed=1226, test_size=0.2, valid_size=0.2):

    check_folder('./data/')

    if split is None:
        split = len(c0)

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


def taiwan_loader(batch_size=1, seed=1226, state=1226, test_size=0.2, valid_size=0.2, num_workers=0, pin_memory=True,
                  std=1, c=[], adv=[], adv_valid=True, u='taiwan.csv', counts=False):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # import dataset
    data = pd.read_csv('./data/' + u, header=0)

    X = data.values[:, 1:]
    y = data.values[:, 0]

    X, X_test, y, y_test = train_test_split(np.array(X), np.array(y), test_size=test_size, random_state=state)
    X, X_valid, y, y_valid = train_test_split(np.array(X), np.array(y), test_size=valid_size, random_state=state)

    train_loader, valid_loader, test_loader = getLoaders(X, X_valid, X_test, y, y_valid, y_test, batch_size, seed,
                                                         num_workers, pin_memory, std, c, adv, adv_valid, counts, True)

    return train_loader, valid_loader, test_loader


def replace_str_cols(co, col_names):
    cols = []
    for cn in co:
        cols.extend([c for c in range(len(col_names)) if col_names[c].startswith(cn)])
    return cols


with open('ni_cols_orig.txt', 'r') as f:
    cols_orig = f.read().split('\n')


def ni_loader(batch_size=1, seed=1226, state=1226, train_size=1, valid_size=0.2, num_workers=0, pin_memory=True, std=1,
              c=[], adv=[], adv_valid=True, classes=2, plus=True, counts=False):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load Data
    if plus:
        filename2 = './data/NSL-KDD/KDDTrain+.txt'
        filename3 = './data/NSL-KDD/KDDTest+.txt'
    else:
        u = 'http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz'
        filename2 = download(u)
        u = 'http://kdd.ics.uci.edu/databases/kddcup99/corrected.gz'
        filename3 = download(u)

    # import dataset
    train = pd.read_csv(filename2, header=None)
    test = pd.read_csv(filename3, header=None)

    if plus:
        train.drop(columns=train.columns[-1], axis=1, inplace=True)
        test.drop(columns=test.columns[-1], axis=1, inplace=True)

    # transform to super-categories
    i = train.shape[1] - 1
    if classes == 2:
        """
        dic = {'normal.': 'normal', 'land.': 'dos', 'pod.': 'dos', 'teardrop.': 'dos', 'back.': 'dos',
               'neptune.': 'dos', 'smurf.': 'dos'}
        """
        dic = {key: 'attack' for key in train[i].unique()}
        dic['normal'] = 'normal'
    elif classes == 5:
        dic = {'normal.': 'normal', 'nmap.': 'probing', 'portsweep.': 'probing', 'ipsweep.': 'probing',
               'satan.': 'probing', 'land.': 'dos', 'pod.': 'dos', 'teardrop.': 'dos', 'back.': 'dos',
               'neptune.': 'dos', 'smurf.': 'dos', 'spy.': 'r2l', 'phf.': 'r2l', 'multihop.': 'r2l',
               'ftp_write.': 'r2l', 'imap.': 'r2l', 'warezmaster.': 'r2l', 'guess_passwd.': 'r2l',
               'buffer_overflow.': 'u2r', 'rootkit.': 'u2r', 'loadmodule.': 'u2r', 'perl.': 'u2r'}
    else:
        raise Exception('Invalid number of classes')
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

    if train_size < 1:
        X, _, y, _ = train_test_split(np.array(X), np.array(y), test_size=train_size, random_state=state)
    X, X_valid, y, y_valid = train_test_split(np.array(X), np.array(y), test_size=valid_size, random_state=state)

    train_loader, valid_loader, test_loader = getLoaders(X, X_valid, X_test, y, y_valid, y_test, batch_size, seed,
                                                         num_workers, pin_memory, std, c, adv, adv_valid, counts, True)

    return train_loader, valid_loader, test_loader


def ibm_loader(batch_size=1, seed=1226, state=1226, test_size=0.2, valid_size=0.2, num_workers=0, pin_memory=True,
               std=1, undersample=None, c=[], adv=[], adv_valid=True, verbose=False):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load Data
    filename = './data/IBM.csv'

    # import dataset
    df = pd.read_csv(filename, header=None)
    df.drop(columns=df.columns[0], axis=1, inplace=True)
    if verbose:
        print(df.shape)

    # undersample dominant class
    ccol = df.columns[-1]
    if undersample is not None:
        df_0 = df[~df[ccol]]
        df_1 = df[df[ccol]]
        df_0_under = df_0.sample(undersample * df_1.shape[0], random_state=seed)
        df = pd.concat([df_0_under, df_1], axis=0)
        if verbose:
            print(df.shape)
    df[ccol] = df[ccol].replace({True: 1, False: 0})

    # split classes & features
    X = df.values[:, :-1]
    y = df.values[:, -1]
    if verbose:
        print(X.shape)

    # one-hot encode categorical variables
    X = pd.DataFrame(X)
    int_vals = np.array([X[col].apply(float.is_integer).all() for col in X.columns])
    nunique = np.array(X.nunique())
    cols = np.where(np.logical_and(np.logical_and(2 < nunique, nunique < 10), int_vals))[0]
    X = pd.get_dummies(X, columns=X.columns[cols])  # convert objects to one-hot encoding
    if verbose:
        print(X.shape)
    """
    for col in X.columns:
        print(col)
    # """
    X = X.to_numpy()

    X, X_test, y, y_test = train_test_split(np.array(X), np.array(y), test_size=test_size, random_state=state)
    X, X_valid, y, y_valid = train_test_split(np.array(X), np.array(y), test_size=valid_size, random_state=state)

    train_loader, valid_loader, test_loader = getLoaders(X, X_valid, X_test, y, y_valid, y_test, batch_size, seed,
                                                         num_workers, pin_memory, std, c, adv, adv_valid, verbose, True)

    return train_loader, valid_loader, test_loader
