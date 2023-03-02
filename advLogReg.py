from floaders import *
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)


e0 = 14
s1 = 10
epochs = 100
inner = 10
test_size, valid_size = 0.2, 0.2
random_seed = 1226
model = LogisticRegression(max_iter=inner)


def alpha(k):
    return 1/(k+1)


# Load Data
u = 'https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz'
filename2 = download(u)

# import dataset
data = pd.read_csv(filename2, header=None)

X = data.values[:, :-1]
y = data.values[:, -1] - 1

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

X = np.concatenate((X[:, :e0], X[:, s1:]), axis=1)
X_valid = np.concatenate((X_valid[:, :e0], X_valid[:, s1:]), axis=1)
X_test = np.concatenate((X_test[:, :e0], X_test[:, s1:]), axis=1)


def adversary(model, X):
    temp = 1 / (np.cosh(.5 * np.inner(X, model.coef_)) ** 2)
    grad = -.25 * np.dot(temp, model.coef_)
    X[:, :e0] += alpha(j) * grad[:, :e0]
    return X


loss = []
for i in range(epochs):
    model.fit(X, y)
    for j in range(inner):
        X = adversary(model, X)
        X_valid = adversary(model, X_valid)
        X_test = adversary(model, X_test)
    loss.append(model.score(X_valid, y_valid))

check_folder('./data')
np.savetxt("./data/advLogReg.csv", X, delimiter=",")
np.savetxt("./data/advLogReg_valid.csv", X_valid, delimiter=",")
np.savetxt("./data/advLogReg_test.csv", X_test, delimiter=",")
np.savetxt("./data/advLogReg_y.csv", y, delimiter=",")
np.savetxt("./data/advLogReg_y_valid.csv", y_valid, delimiter=",")
np.savetxt("./data/advLogReg_y_test.csv", y_test, delimiter=",")

check_folder('./plots')
plt.plot(loss)
plt.title('Validation Accuracy')
plt.savefig('./plots/advLogReg.png')
plt.clf()
plt.close()
