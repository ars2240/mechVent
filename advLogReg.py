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
# c0 = [1, 6, 8]
c0 = [6, 20, 22, 25, 32, 40, 49]
print('client 0: {0}'.format(c0))
# c1 = [*range(6, 9), *range(10, 54)]
# c1 = [4, 5, 9]
c1 = [9, 21, 28, 31, 38, 39, 41]
print('client 1: {0}'.format(c1))
shared = [x for x in range(0, 54) if x not in c0 and x not in c1]
print('shared: {0}'.format(shared))
ag_ignore = True
epochs = 300
inner = 10
test_size, valid_size = 0.2, 0.2
random_seed = 1226
model = LogisticRegression(max_iter=inner)
head = 'advLogReg3AdamIgn'
adv_opt = 'adam'
adv_beta = (0.9, 0.999)
adv_eps = 1e-8
alpha = 0.001


"""
def alpha(k):
    return 1/(k+1)
"""

adv = [*range(len(c0), len(c0)+len(shared))]
c0.extend(shared)
if not ag_ignore:
    c1.extend(shared)

# Load Data
u = 'https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz'
filename2 = download(u)

# import dataset
data = pd.read_csv(filename2, header=None)

X = data.values[:, :-1]
y = data.values[:, -1] - 1

enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(y.reshape(-1, 1))

X, X_test, y, y_test = train_test_split(np.array(X), np.array(y), test_size=test_size, random_state=random_seed)

X = X.reshape((X.shape[0], 54))
X_test = X_test.reshape((X_test.shape[0], 54))

X, X_valid, y, y_valid = train_test_split(np.array(X), np.array(y), test_size=valid_size, random_state=random_seed)

# normalize data
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

X = np.concatenate((X[:, c0], X[:, c1]), axis=1)
X_valid = np.concatenate((X_valid[:, c0], X_valid[:, c1]), axis=1)
X_test = np.concatenate((X_test[:, c0], X_test[:, c1]), axis=1)

if ag_ignore:
    X_ag = X[:, adv]
    X_ag_valid = X_valid[:, adv]
    X_ag_test = X_test[:, adv]


def adversary(model, X, y, j):
    w = model.coef_
    b = model.intercept_
    y = enc.transform(y.reshape(-1, 1)).toarray()
    e = np.minimum(np.inner(X, w)+b, 709)
    grad = -np.matmul(y/(1+np.exp(e)), w)
    X[:, adv] += alpha(j) * grad[:, adv]
    return X


def adversary_adam(model, X, y, j, m, v):
    w = model.coef_
    b = model.intercept_
    y = enc.transform(y.reshape(-1, 1)).toarray()
    e = np.minimum(np.inner(X, w)+b, 709)
    grad = -np.matmul(y/(1+np.exp(e)), w)
    m = adv_beta[0] * m + (1 - adv_beta[0]) * grad
    v = adv_beta[1] * v + (1 - adv_beta[1]) * grad ** 2
    mhat = m / (1.0 - adv_beta[0] ** (j + 1))
    vhat = v / (1.0 - adv_beta[1] ** (j + 1))
    grad = mhat / (np.sqrt(vhat) + adv_eps)
    X[:, adv] += alpha * grad[:, adv]
    return X, m, v


loss, lossH = [], []
best_acc = 1
best_model = None
X_best, X_valid_best, X_test_best = X, X_valid, X_test
m, v, m_valid, v_valid, m_test, v_test = 0, 0, 0, 0, 0, 0
for i in range(epochs):
    model.fit(X, y)
    lossH.append(model.score(X_valid, y_valid))
    for j in range(inner):
        if adv_opt.lower() == 'sgd':
            X = adversary(model, X, y, j)
            X_valid = adversary(model, X_valid, y_valid, j)
            X_test = adversary(model, X_test, y_test, j)
        elif adv_opt.lower() == 'adam':
            X, m, v = adversary_adam(model, X, y, j, m, v)
            X_valid, m_valid, v_valid = adversary_adam(model, X_valid, y_valid, j, m_valid, v_valid)
            X_test, m_test, v_test = adversary_adam(model, X_test, y_test, j, m_test, v_test)
    l = model.score(X_valid, y_valid)
    if l < best_acc:
        best_acc = l
        best_model = model
        X_best, X_valid_best, X_test_best = X, X_valid, X_test
    loss.append(l)

check_folder('./logs')
np.savetxt("./logs/" + head + "_coef.csv", model.coef_, delimiter=",")
np.savetxt("./logs/" + head + "_intercept.csv", model.intercept_, delimiter=",")
model_fi = permutation_importance(model, X, y)
np.savetxt("./logs/" + head + "_importanceAdv.csv", model_fi['importances_mean'], delimiter=",")

np.savetxt("./logs/" + head + "_best_coef.csv", best_model.coef_, delimiter=",")
np.savetxt("./logs/" + head + "_best_intercept.csv", best_model.intercept_, delimiter=",")
model_fi = permutation_importance(best_model, X, y)
np.savetxt("./logs/" + head + "_best_importanceAdv.csv", model_fi['importances_mean'], delimiter=",")

if ag_ignore:
    X = np.concatenate((X, X_ag), axis=1)
    X_valid = np.concatenate((X_valid, X_ag_valid), axis=1)
    X_test = np.concatenate((X_test, X_ag_test), axis=1)

check_folder('./data')
np.savetxt("./data/" + head + ".csv", X, delimiter=",")
np.savetxt("./data/" + head + "_valid.csv", X_valid, delimiter=",")
np.savetxt("./data/" + head + "_test.csv", X_test, delimiter=",")
np.savetxt("./data/" + head + "_y.csv", y, delimiter=",")
np.savetxt("./data/" + head + "_y_valid.csv", y_valid, delimiter=",")
np.savetxt("./data/" + head + "_y_test.csv", y_test, delimiter=",")

if ag_ignore:
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
plt.plot(loss)
plt.title('Validation Accuracy')
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.savefig('./plots/' + head + '.png')
plt.clf()
plt.close()


