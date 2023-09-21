import numpy as np
from floaders import *
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

c0 = [0, 1, 6, 7, 14, 16, 17, 19, 25, 26, 27, 30, 31, 32, 33, 34, 35, 36, 37, *range(38, 41)]
print('client 0: {0}'.format(c0))
c1 = [2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 15, 18, 20, 21, 23, 24, 28, 29, *range(41, 122)]
print('client 1: {0}'.format(c1))
shared = [x for x in range(0, 122) if x not in c0 and x not in c1]
print('shared: {0}'.format(shared))
fl = 'none'  # none, horizontal, or vertical
plus = True
adv_valid = True
rand_init = True
epochs = 1
inner = 100
fill = 0
test_size, valid_size = 0.2, 0.2
state = 1226
model = LogisticRegression(max_iter=inner)
modelC = LogisticRegression(max_iter=inner)
head = 'NI+Share1'
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

"""
print('Train classes: {0}'.format(np.sum(y)/len(y)))
print('Valid classes: {0}'.format(np.sum(y_valid)/len(y_valid)))
print('Test classes: {0}'.format(np.sum(y_test)/len(y_test)))
"""

# normalize data
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)


def horizontalize(X):
    X0, X1 = X.copy(), X.copy()
    X0[:, c1], X1[:, c0] = fill, fill
    if rand_init:
        X0[:, shared] = np.random.normal(size=(X0.shape[0], len(shared)))
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
    X = np.concatenate((X[:, c0], X[:, c1]), axis=1)
    X_valid = np.concatenate((X_valid[:, c0], X_valid[:, c1]), axis=1)
    X_test = np.concatenate((X_test[:, c0], X_test[:, c1]), axis=1)

if fl.lower() == 'none':
    X_ag = X[:, adv]
    X_ag_valid = X_valid[:, adv]
    X_ag_test = X_test[:, adv]

if rand_init and fl.lower() != 'horizontal':
    X[:, adv] = np.random.normal(size=(X.shape[0], len(adv)))
    if adv_valid:
        X_valid[:, adv] = np.random.normal(size=(X_valid.shape[0], len(adv)))
        X_test[:, adv] = np.random.normal(size=(X_test.shape[0], len(adv)))


def adversary(model, X, y, j):
    w = model.coef_
    b = model.intercept_
    e = np.minimum(np.inner(X, w)+b, 709)
    grad = -np.matmul(y.reshape(-1, 1)/(1+np.exp(e)), w)
    if fl.lower() == 'horizontal':
        st = int(X.shape[0] / 2)
        X[:st, shared] += alpha * grad[:st, shared]
    else:
        X[:, adv] += alpha * grad[:, adv]
    return X


def adversary_adam(model, X, y, j, m, v):
    w = model.coef_
    b = model.intercept_
    e = np.minimum(np.inner(X, w)+b, 709)
    grad = -np.matmul(y.reshape(-1, 1)/(1+np.exp(e)), w)
    m = adv_beta[0] * m + (1 - adv_beta[0]) * grad
    v = adv_beta[1] * v + (1 - adv_beta[1]) * grad ** 2
    mhat = m / (1.0 - adv_beta[0] ** (j + 1))
    vhat = v / (1.0 - adv_beta[1] ** (j + 1))
    grad = mhat / (np.sqrt(vhat) + adv_eps)
    if fl.lower() == 'horizontal':
        st = int(X.shape[0] / 2)
        X[:st, shared] += alpha * grad[:st, shared]
    else:
        X[:, adv] += alpha * grad[:, adv]
    return X, m, v


loss, lossH, lossC = [], [], []
best_acc = 1
best_model = None
X_best, X_valid_best, X_test_best = X.copy(), X_valid.copy(), X_test.copy()
m, v, m_valid, v_valid, m_test, v_test = 0, 0, 0, 0, 0, 0
for i in range(epochs):
    model.fit(X, y)
    if adv_valid:
        lossH.append(model.score(X_valid, y_valid))
    if len(shared) > 0:
        for j in range(inner):
            if adv_opt.lower() == 'sgd':
                X = adversary(model, X, y, j)
                if adv_valid:
                    X_valid = adversary(model, X_valid, y_valid, j)
                    X_test = adversary(model, X_test, y_test, j)
            elif adv_opt.lower() == 'adam':
                X, m, v = adversary_adam(model, X, y, j, m, v)
                if adv_valid:
                    X_valid, m_valid, v_valid = adversary_adam(model, X_valid, y_valid, j, m_valid, v_valid)
                    X_test, m_test, v_test = adversary_adam(model, X_test, y_test, j, m_test, v_test)
    l = model.score(X_valid, y_valid)
    if l < best_acc:
        best_acc = l
        best_model = model
        X_best, X_valid_best, X_test_best = X.copy(), X_valid.copy(), X_test.copy()
    loss.append(l)
    if fl.lower() == 'none':
        modelC.fit(np.concatenate((X, X_ag), axis=1), y)
        lc = modelC.score(np.concatenate((X_valid, X_ag_valid), axis=1), y_valid)
        lossC.append(lc)


print('Train Accuracy: {0}'.format(best_model.score(X, y)))
print('Valid Accuracy: {0}'.format(best_acc))
print('Test Accuracy: {0}'.format(best_model.score(X_test, y_test)))

check_folder('./logs')
np.savetxt("./logs/" + head + "_coef.csv", model.coef_, delimiter=",")
np.savetxt("./logs/" + head + "_intercept.csv", model.intercept_, delimiter=",")
model_fi = permutation_importance(model, X, y)
np.savetxt("./logs/" + head + "_importanceAdv.csv", model_fi['importances_mean'], delimiter=",")

np.savetxt("./logs/" + head + "_best_coef.csv", best_model.coef_, delimiter=",")
np.savetxt("./logs/" + head + "_best_intercept.csv", best_model.intercept_, delimiter=",")
model_fi = permutation_importance(best_model, X, y)
np.savetxt("./logs/" + head + "_best_importanceAdv.csv", model_fi['importances_mean'], delimiter=",")

if fl.lower() == 'none':
    X = np.concatenate((X, X_ag), axis=1)
    X_valid = np.concatenate((X_valid, X_ag_valid), axis=1)
    X_test = np.concatenate((X_test, X_ag_test), axis=1)

"""
check_folder('./data')
np.savetxt("./data/" + head + ".csv", X, delimiter=",")
np.savetxt("./data/" + head + "_valid.csv", X_valid, delimiter=",")
np.savetxt("./data/" + head + "_test.csv", X_test, delimiter=",")
np.savetxt("./data/" + head + "_y.csv", y, delimiter=",")
np.savetxt("./data/" + head + "_y_valid.csv", y_valid, delimiter=",")
np.savetxt("./data/" + head + "_y_test.csv", y_test, delimiter=",")
"""

if fl.lower() == 'none':
    X_best = np.concatenate((X_best, X_ag), axis=1)
    X_valid_best = np.concatenate((X_valid_best, X_ag_valid), axis=1)
    X_test_best = np.concatenate((X_test_best, X_ag_test), axis=1)

check_folder('./data')
np.savetxt("./data/" + head + "_best.csv", X_best, delimiter=",")
np.savetxt("./data/" + head + "_best_valid.csv", X_valid_best, delimiter=",")
np.savetxt("./data/" + head + "_best_test.csv", X_test_best, delimiter=",")
np.savetxt("./data/" + head + "_best_y.csv", y, delimiter=",")
np.savetxt("./data/" + head + "_best_y_valid.csv", y_valid, delimiter=",")
np.savetxt("./data/" + head + "_best_y_test.csv", y_test, delimiter=",")

check_folder('./plots')
plt.plot(loss, label='post-Adv')
if adv_valid:
    plt.plot(lossH, label='pre-Adv')
if fl.lower() == 'none':
    plt.plot(lossC, label='combined')
plt.title('Validation Accuracy')
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
if adv_valid or fl.lower() == 'none':
    plt.legend()
plt.savefig('./plots/' + head + '.png')
plt.clf()
plt.close()


