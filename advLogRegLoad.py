import numpy as np
from floaders import *
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)


# c0 = [*range(0, 6), 9, *range(14, 54)]
c0 = [1, 6, 8]
# c0 = [6, 20, 22, 25, 32, 40, 49]
print('client 0: {0}'.format(c0))
# c1 = [*range(6, 9), *range(10, 54)]
c1 = [4, 5, 9]
# c1 = [9, 21, 28, 31, 38, 39, 41]
print('client 1: {0}'.format(c1))
shared = [x for x in range(0, 54) if x not in c0 and x not in c1]
print('shared: {0}'.format(shared))
fl = 'horizontal'  # none, horizontal, or vertical
adv_valid = False
inner = 1000
fill = 0
test_size, valid_size = 0.2, 0.2
random_seed = 1226
model = LogisticRegression(max_iter=inner)
head = 'advLogReg2AdamIgnRandInit_best'


"""
def alpha(k):
    return 1/(k+1)
"""

nf = len(c0)+len(c1)+len(shared)
split = len(c0)+len(shared)
adv = [*range(len(c0), split)]
if fl.lower() != 'horizontal':
    c0.extend(shared)
if fl.lower() == 'vertical':
    c1.extend(shared)

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
    u = 'https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz'
    filename2 = download(u)

    # import dataset
    data = pd.read_csv(filename2, header=None)

    Xg = data.values[:, :-1]
    Xg = Xg.reshape((Xg.shape[0], 54))
    yg = data.values[:, -1] - 1

    Xg, X_test, yg, y_test = train_test_split(np.array(Xg), np.array(yg), test_size=test_size, random_state=random_seed)
    Xg, X_valid, yg, y_valid = train_test_split(np.array(Xg), np.array(yg), test_size=valid_size,
                                                random_state=random_seed)

    # normalize data
    scaler = StandardScaler()
    scaler.fit(Xg)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)


def horizontalize(X):
    X0, X1 = np.zeros((X.shape[0], nf)), np.zeros((X.shape[0], nf))
    X0[:, c0] = X[:, :len(c0)]
    X0[:, shared] = X[:, len(c0):split]
    X1[:, c1] = X[:, split:nf]
    X1[:, shared] = X[:, nf:]
    return np.concatenate((X0, X1), axis=0)


if fl.lower() == 'horizontal':
    X = horizontalize(X)
    y = np.concatenate((y, y), axis=0)
    if adv_valid:
        X_valid = horizontalize(X_valid)
        y_valid = np.concatenate((y_valid, y_valid), axis=0)
        X_test = horizontalize(X_test)
        y_test = np.concatenate((y_test, y_test), axis=0)
else:
    X = np.concatenate((X[:, :split], X[:, split:]), axis=1)
    X_valid = np.concatenate((X_valid[:, :split], X_valid[:, split:]), axis=1)
    X_test = np.concatenate((X_test[:, :split], X_test[:, split:]), axis=1)


model.fit(X, y)
val_acc = model.score(X_valid, y_valid)

print('Validation accuracy: {0}'.format(val_acc))



