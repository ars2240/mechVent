from advLogReg import advLogReg
import numpy as np
from floaders import *
from sklearn.linear_model import LogisticRegression

c0 = [*range(0, 54)]
print('client 0: {0}'.format(c0))
c1 = []
print('client 1: {0}'.format(c1))
shared = [x for x in range(0, 54) if x not in c0 and x not in c1]
print('shared: {0}'.format(shared))
fl = 'none'  # none, horizontal, or vertical
adv_valid = False
rand_init = True
epochs = 100
inner = 100
fill = 0
test_size, valid_size = 0.2, 0.2
random_seed = 1226
model = LogisticRegression(max_iter=inner)
head = 'advLogReg2AdamRandInitShare0'
adv_opt = 'adam'
adv_beta = (0.9, 0.999)
adv_eps = 1e-8
alpha = 0.001


adv = [*range(len(c0), len(c0)+len(shared))]

# Load Data
u = 'https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz'
filename2 = download(u)

# import dataset
data = pd.read_csv(filename2, header=None)

X = data.values[:, :-1]
X = X.reshape((X.shape[0], 54))
y = data.values[:, -1] - 1

X, X_test, y, y_test = train_test_split(np.array(X), np.array(y), test_size=test_size, random_state=random_seed)
X, X_valid, y, y_valid = train_test_split(np.array(X), np.array(y), test_size=valid_size, random_state=random_seed)

advLogReg(X, X_valid, X_test, y, y_valid, y_test, fl, adv_valid, rand_init, epochs, inner, fill, adv_opt, adv_beta,
          adv_eps, alpha, c0, c1, shared, adv, model, head)


