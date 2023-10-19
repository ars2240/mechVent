from advLogReg import advLogReg
import numpy as np
from floaders import *
from sklearn.linear_model import LogisticRegression
from itertools import chain

sh = 41
fl = 'none'  # none, horizontal, or vertical
plus = True
adv_valid = True
rand_init = True
epochs = 100
inner = 100
fill = 0
test_size, valid_size = 0.2, 0.2
state = 1226
model = LogisticRegression(max_iter=inner)
modelC = LogisticRegression(max_iter=inner)
head = 'NI+Share'
adv_opt = 'adam'
adv_beta = (0.9, 0.999)
adv_eps = 1e-8
alpha = 0.001
classes = 2

np.random.seed(state)

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

# transform to supercategories
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

X, X_valid, y, y_valid = train_test_split(np.array(X), np.array(y), test_size=valid_size, random_state=state)

for sh in range(1, 42, 10):
    if sh == 1:
        c = [[16, 19, 25, 26], [5, 8, 12, 24], [3, 9, 29, *range(41, 111)], [7, 17, 32, 35],
             [14, 36, 37, *range(38, 41)],
             [4, 15, 20, 23], [2, 13, 18, 28], [1, 6, 30, 31], [0, 27, 33, 34], [10, 11, 21, *range(111, 122)]]
    elif sh == 11:
        c = [[16, 25, 26], [5, 8, 12], [3, 9, *range(41, 111)], [7, 17, 35], [14, 37, *range(38, 41)], [15, 20, 23],
             [2, 13, 28], [1, 6, 31], [0, 27, 34], [10, 11, 21]]
    elif sh == 21:
        c = [[16, 25], [5, 12], [3, 9], [7, 17], [14, 37], [15, 23], [2, 13], [1, 31], [0, 34], [10, 11]]
    elif sh == 31:
        c = [[16], [5], [3], [17], [14], [15], [13], [1], [0], [10]]
    elif sh == 41:
        c = [[], [], [], [], [], [], [], [], [], []]
    else:
        raise Exception('Number of shared features not implemented.')
    shared = [x for x in range(0, 122) if x not in chain(*c)]
    print('shared: {0}'.format(shared))

    adv = shared

    advLogReg(X, X_valid, X_test, y, y_valid, y_test, fl, adv_valid, rand_init, epochs, inner, fill, adv_opt, adv_beta,
              adv_eps, alpha, c0, c1, shared, adv, model, head + str(sh))


