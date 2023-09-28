import numpy as np
from floaders import *
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

sh = 1
if sh == 1:
    c0 = [*range(348, 352), 32, 69, 99, 114, 196, 286, 324, *range(352, 358), 45, 46, 58, 259, 209, 220, 312, 1, 12, 30,
          61, 83, 89, 113, 145, 232, 235, 272, 281, 300, 311, 322, 34, 49, 202, 212, 217, 233, 269, 270, 271, 287, 305,
          330, 191, 226, 228, 273, 298, 6, 82, 92, 225, 319, 40, 47, 75, 77, 78, 117, 248, 267, 123, 208, 67, 219, 304,
          218, *range(335, 340), 13, 25, 35, 38, 86, 107, 121, 170, 172, 174, 204, 206, 207, 210, 211, 139, 294, 129,
          130, 131, 132, 133, 134, 135, 136, 137, 138, 90, 164, 177, 180, 181, 182, 183, 184, 195, 198, 199, 200, 21,
          33, 120, 122, 152, 153, 154, 155, 95, 252, 42, 240, 106, 323, 87, 246, 14, 186, 16, 274, 22, 296, 282, 8, 241,
          277, 24, 27, 243, 234, 276, 283, 105, 229, 316, 93, 140, 262, 260, 279, 54, 146, 318, 265, 148, 150, 151, 166,
          244, 10, 242, 308, 28, 303, 230, 194, 268, 5, 63, 79, 257, 141, 56]
    c1 = [328, *range(345, 348), *range(358, 363), 310, 315, 11, 64, 81, 94, 108, 109, 179, 185, 213, 247, 317, 325,
          327, 52, 171, 173, 175, 188, 50, 51, 72, 104, 110, 111, 115, 118, 119, 124, 142, 143, 144, 147, 149, 165, 169,
          176, 192, 201, 239, 295, 321, 329, 334, *range(340, 345), 70, 102, 289, 291, 214, 215, 216, 224, 331, 19, 73,
          80, 96, 101, 178, 293, 36, 231, 306, 307, 309, 9, 55, 66, 74, 116, 4, 85, 261, 17, 65, 264, 41, 197, 297, 31,
          156, 157, 158, 159, 160, 161, 162, 163, 203, 205, 284, 333, 256, 290, 98, 100, 125, 126, 127, 128, 187, 222,
          236, 26, 37, 275, 313, 292, 3, 76, 88, 112, 258, 314, 332, 43, 326, 280, 97, 0, 91, 221, 193, 253, 18, 237,
          288, 301, 68, 190, 15, 223, 249, 62, 250, 285, 48, 245, 263, 84, 299, 302, 103, 20, 39, 278, 29, 60, 254, 251,
          23, 53, 255, 227, 44, 59, 57, 320, 167, 2, 71, 189, 7, 168, 238]
elif sh == 85:
    c0 = [*range(348, 352), 32, 69, 99, 114, 196, 286, 324, *range(352, 358), 45, 46, 58, 259, 209, 220, 312, 1, 12, 30,
          61, 83, 89, 113, 145, 232, 235, 272, 281, 300, 311, 322, 34, 49, 202, 212, 217, 233, 269, 270, 271, 287, 305,
          330, 191, 226, 228, 273, 298, 6, 82, 92, 225, 319, 40, 47, 75, 77, 78, 117, 248, 267, 123, 208, 67, 219, 304,
          218, *range(335, 340), 13, 25, 35, 38, 86, 107, 121, 170, 172, 174, 204, 206, 207, 210, 211, 139, 294, 129,
          130, 131, 132, 133, 134, 135, 136, 137, 138, 90, 164, 177, 180, 181, 182, 183, 184, 195, 198, 199, 200, 21,
          33, 120, 122, 152, 153, 154, 155, 95, 252, 42, 240, 106, 323, 87, 246, 14, 186, 16, 274, 22]
    c1 = [328, *range(345, 348), *range(358, 363), 310, 315, 11, 64, 81, 94, 108, 109, 179, 185, 213, 247, 317, 325,
          327, 52, 171, 173, 175, 188, 50, 51, 72, 104, 110, 111, 115, 118, 119, 124, 142, 143, 144, 147, 149, 165, 169,
          176, 192, 201, 239, 295, 321, 329, 334, *range(340, 345), 70, 102, 289, 291, 214, 215, 216, 224, 331, 19, 73,
          80, 96, 101, 178, 293, 36, 231, 306, 307, 309, 9, 55, 66, 74, 116, 4, 85, 261, 17, 65, 264, 41, 197, 297, 31,
          156, 157, 158, 159, 160, 161, 162, 163, 203, 205, 284, 333, 256, 290, 98, 100, 125, 126, 127, 128, 187, 222,
          236, 26, 37, 275, 313, 292, 3, 76, 88, 112, 258, 314, 332, 43, 326, 280, 97, 0, 91, 221, 193]
elif sh == 171:
    c0 = [*range(348, 352), 32, 69, 99, 114, 196, 286, 324, *range(352, 358), 45, 46, 58, 259, 209, 220, 312, 1, 12, 30,
          61, 83, 89, 113, 145, 232, 235, 272, 281, 300, 311, 322, 34, 49, 202, 212, 217, 233, 269, 270, 271, 287, 305,
          330, 191, 226, 228, 273, 298, 6, 82, 92, 225, 319, 40, 47, 75, 77, 78, 117, 248, 267, 123, 208, 67, 219, 304,
          218, *range(335, 340), 13, 25, 35, 38, 86, 107, 121, 170, 172, 174, 204, 206, 207, 210, 211, 139, 294]
    c1 = [328, *range(345, 348), *range(358, 363), 310, 315, 11, 64, 81, 94, 108, 109, 179, 185, 213, 247, 317, 325,
          327, 52, 171, 173, 175, 188, 50, 51, 72, 104, 110, 111, 115, 118, 119, 124, 142, 143, 144, 147, 149, 165, 169,
          176, 192, 201, 239, 295, 321, 329, 334, *range(340, 345), 70, 102, 289, 291, 214, 215, 216, 224, 331, 19, 73,
          80, 96, 101, 178, 293, 36, 231, 306, 307, 309, 9, 55, 66, 74, 116, 4, 85, 261, 17, 65, 264, 41, 197, 297, 31]
elif sh == 255:
    c0 = [*range(348, 352), 32, 69, 99, 114, 196, 286, 324, *range(352, 358), 45, 46, 58, 259, 209, 220, 312, 1, 12, 30,
          61, 83, 89, 113, 145, 232, 235, 272, 281, 300, 311, 322, 34, 49, 202, 212, 217, 233, 269, 270, 271, 287, 305,
          330]
    c1 = [328, *range(345, 348), *range(358, 363), 310, 315, 11, 64, 81, 94, 108, 109, 179, 185, 213, 247, 317, 325,
          327, 52, 171, 173, 175, 188, 50, 51, 72, 104, 110, 111, 115, 118, 119, 124, 142, 143, 144, 147, 149, 165, 169,
          176, 192, 201]
elif sh == 341:
    c0 = []
    c1 = []
else:
    raise Exception('Number of shared features not implemented.')
c0.sort()
c1.sort()
print('client 0: {0}'.format(c0))
print('client 1: {0}'.format(c1))
shared = [x for x in range(0, 363) if x not in c0 and x not in c1]
print('shared: {0}'.format(shared))
fl = 'none'  # none, horizontal, or vertical
plus = True
adv_valid = True
rand_init = True
epochs = 1
inner = 100
fill = 0
test_size, valid_size = 0.2, 0.2
seed = 1226
model = LogisticRegression(max_iter=inner)
modelC = LogisticRegression(max_iter=inner)
head = 'IBMU4_Sh' + str(sh)
adv_opt = 'adam'
adv_beta = (0.9, 0.999)
adv_eps = 1e-8
alpha = 0.001
undersample = 4

adv = [*range(len(c0), len(c0)+len(shared))]
if fl.lower() != 'horizontal':
    c0.extend(shared)
if fl.lower() == 'vertical':
    c1.extend(shared)

np.random.seed(seed)
torch.manual_seed(seed)

# Load Data
filename = './data/IBM.csv'

# import dataset
df = pd.read_csv(filename, header=None)
df.drop(columns=df.columns[0], axis=1, inplace=True)

# undersample dominant class
ccol = df.columns[-1]
if undersample is not None:
    df_0 = df[~df[ccol]]
    df_1 = df[df[ccol]]
    df_0_under = df_0.sample(undersample * df_1.shape[0], random_state=seed)
    df = pd.concat([df_0_under, df_1], axis=0)
df[ccol] = df[ccol].replace({True: 1, False: 0})

# split classes & features
X = df.values[:, :-1]
y = df.values[:, -1]

# one-hot encode categorical variables
X = pd.DataFrame(X)
int_vals = np.array([X[col].apply(float.is_integer).all() for col in X.columns])
nunique = np.array(X.nunique())
cols = np.where(np.logical_and(np.logical_and(2 < nunique, nunique < 10), int_vals))[0]
X = pd.get_dummies(X, columns=X.columns[cols])  # convert objects to one-hot encoding
X = X.to_numpy()

X, X_test, y, y_test = train_test_split(np.array(X), np.array(y), test_size=test_size, random_state=seed)
X, X_valid, y, y_valid = train_test_split(np.array(X), np.array(y), test_size=valid_size, random_state=seed)

"""
print('Train classes: {0}'.format(np.sum(y)/len(y)*100))
print('Valid classes: {0}'.format(np.sum(y_valid)/len(y_valid)*100))
print('Test classes: {0}'.format(np.sum(y_test)/len(y_test)*100))
# """

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


print('Train Accuracy: {0}'.format(best_model.score(X, y)*100))
print('Valid Accuracy: {0}'.format(best_acc*100))
print('Test Accuracy: {0}'.format(best_model.score(X_test, y_test)*100))

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


