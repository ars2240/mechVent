import numpy as np
from floaders import *
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

sh = 41
if sh == 1:
    c0 = [0, 1, 6, 7, 14, 16, 17, 19, 25, 26, 27, 30, 31, 32, 33, 34, 35, 36, 37, *range(38, 41)]
    c1 = [2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 15, 18, 20, 21, 23, 24, 28, 29, *range(41, 122)]
elif sh == 11:
    c0 = [0, 1, 6, 7, 14, 16, 17, 25, 26, 27, 31, 34, 35, 37, *range(38, 41)]
    c1 = [2, 3, 5, 8, 9, 10, 11, 12, 13, 15, 20, 21, 23, 28, *range(41, 111)]
elif sh == 21:
    c0 = [0, 1, 7, 14, 16, 17, 25, 31, 34, 37]
    c1 = [2, 3, 5, 9, 10, 11, 12, 13, 15, 23]
elif sh == 31:
    c0 = [0, 1, 14, 16, 17]
    c1 = [3, 5, 10, 13, 15]
elif sh == 41:
    c0 = []
    c1 = []
else:
    raise Exception('Number of shared features not implemented.')
shared = [x for x in range(0, 122) if x not in c0 and x not in c1]
print('client 0: {0}'.format(c0))
print('client 1: {0}'.format(c1))
print('shared: {0}'.format(shared))
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
head = 'NI+Share' + str(sh)
adv_opt = 'adam'
adv_beta = (0.9, 0.999)
adv_eps = 1e-8
alpha = 0.001
classes = 2

np.random.seed(state)
torch.manual_seed(state)


"""
def alpha(k):
    return 1/(k+1)
"""

adv = [*range(len(c0), len(c0)+len(shared))]
if fl.lower() != 'horizontal':
    c0.extend(shared)
if fl.lower() == 'vertical':
    c1.extend(shared)

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

advLogReg(X, X_valid, X_test, y, y_valid, y_test, fl, adv_valid, rand_init, epochs, inner, fill, adv_opt, adv_beta,
          adv_eps, alpha, c0, c1, shared, adv, model, head)


