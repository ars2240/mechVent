from advLogReg import advLogReg
import numpy as np
from floaders import *
from sklearn.linear_model import LogisticRegression
from itertools import chain


fl = 'horizontal'  # none, horizontal, or vertical
adv_valid = True
rand_init = True
epochs = 100
inner = 10
fill = 0
test_size, valid_size = 0.2, 0.2
state = 1226
model = LogisticRegression(max_iter=inner)
modelC = LogisticRegression(max_iter=inner)
# head = 'advLogReg2AdamRandInitShare0'
head = 'Forest10c_Sh'
adv_opt = 'adam'
adv_beta = (0.9, 0.999)
adv_eps = 1e-8
alpha = 0.001
adv_c = [0, 1, 2]

np.random.seed(state)

# Load Data
u = 'https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz'
filename2 = download(u)

# import dataset
data = pd.read_csv(filename2, header=None)

X = data.values[:, :-1]
X = X.reshape((X.shape[0], 54))
y = data.values[:, -1] - 1

X, X_test, y, y_test = train_test_split(np.array(X), np.array(y), test_size=test_size, random_state=state)
X, X_valid, y, y_valid = train_test_split(np.array(X), np.array(y), test_size=valid_size, random_state=state)

for sh in range(12, 13, 10):
    if sh == 2:
        c = [[6], [9], [4], [8], [1], [5], [*range(14, 54)], [2], [*range(10, 14)], [3]]
    elif sh == 12:
        c = [[], [], [], [], [], [], [], [], [], []]
    else:
        raise Exception('Number of shared features not implemented.')
    shared = [x for x in range(0, 54) if x not in chain(*c)]
    print('shared: {0}'.format(shared))

    # adv = [*range(len(c0), len(c0)+len(shared))]
    adv = shared
    c0, c1 = c, None

    advLogReg(X, X_valid, X_test, y, y_valid, y_test, fl, adv_valid, rand_init, epochs, inner, fill, adv_opt, adv_beta,
              adv_eps, alpha, c0, c1, shared, adv, model, modelC, head + str(sh), adv_c)


