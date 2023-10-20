import numpy as np
from floaders import *
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)


def horizontalize(X, rand_init=True, fill=0, c0=[], c1=[], shared=[]):
    X0, X1 = X.copy(), X.copy()
    X0[:, c1], X1[:, c0] = fill, fill
    if rand_init:
        X0[:, shared] = np.random.normal(size=(X0.shape[0], len(shared)))
    return np.concatenate((X0, X1), axis=0)


def adversary(model, X, y, enc, alpha, shared, adv, fl):
    w = model.coef_
    b = model.intercept_
    y = y.reshape(-1, 1)
    if enc is not None:
        y = enc.transform(y).toarray()
    e = np.minimum(np.inner(X, w)+b, 709)
    grad = -np.matmul(y/(1+np.exp(e)), w)
    if fl.lower() == 'horizontal':
        st = int(X.shape[0] / 2)
        X[:st, shared] += alpha * grad[:st, shared]
    else:
        X[:, adv] += alpha * grad[:, adv]
    return X


def adversary_adam(model, X, y, j, m, v, enc, adv_beta, adv_eps, alpha, shared, adv, fl):
    w = model.coef_
    b = model.intercept_
    y = y.reshape(-1, 1)
    if enc is not None:
        y = enc.transform(y).toarray()
    e = np.minimum(np.inner(X, w)+b, 709)
    grad = -np.matmul(y/(1+np.exp(e)), w)
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


def advLogReg(X, X_valid, X_test, y, y_valid, y_test, fl='none', adv_valid=True, rand_init=True, epochs=100, inner=100,
              fill=0, adv_opt='adam', adv_beta=(0.9, 0.999), adv_eps=1e-8, alpha=0.001, c0=None, c1=None, shared=[],
              adv=[], model=None, modelC=None, head=''):
    if model is None:
        model = LogisticRegression(max_iter=inner)
    if modelC is None:
        modelC = LogisticRegression(max_iter=inner)

    if c0 is not None and fl.lower() != 'horizontal':
        c0.extend(shared)
    if c0 is not None and fl.lower() == 'vertical':
        c1.extend(shared)

    # normalize data
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    multi = y.max() > 1
    if multi:
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(y.reshape(-1, 1))
    else:
        enc = None

    if fl.lower() == 'horizontal':
        X = horizontalize(X, rand_init, fill, c0, c1, shared)
        y = np.concatenate((y, y), axis=0)
        if adv_valid:
            X_valid = horizontalize(X_valid, rand_init, fill, c0, c1, shared)
            y_valid = np.concatenate((y_valid, y_valid), axis=0)
            X_test = horizontalize(X_test, rand_init, fill, c0, c1, shared)
            y_test = np.concatenate((y_test, y_test), axis=0)
    else:
        if c0 is not None and c1 is not None:
            X = np.concatenate((X[:, c0], X[:, c1]), axis=1)
            X_valid = np.concatenate((X_valid[:, c0], X_valid[:, c1]), axis=1)
            X_test = np.concatenate((X_test[:, c0], X_test[:, c1]), axis=1)
        elif fl.lower() == 'vertical':
            X = np.concatenate((X, X[:, adv]), axis=1)
            X_valid = np.concatenate((X_valid, X_valid[:, adv]), axis=1)
            X_test = np.concatenate((X_test, X_test[:, adv]), axis=1)

    if fl.lower() == 'none':
        X_ag = X[:, adv]
        X_ag_valid = X_valid[:, adv]
        X_ag_test = X_test[:, adv]

    if rand_init and fl.lower() != 'horizontal':
        X[:, adv] = np.random.normal(size=(X.shape[0], len(adv)))
        if adv_valid:
            X_valid[:, adv] = np.random.normal(size=(X_valid.shape[0], len(adv)))
            X_test[:, adv] = np.random.normal(size=(X_test.shape[0], len(adv)))

    loss, lossH, lossC = [], [], []
    best_acc = 1
    best_model = None
    X_best, X_valid_best, X_test_best = X.copy(), X_valid.copy(), X_test.copy()
    m, v, m_valid, v_valid, m_test, v_test = 0, 0, 0, 0, 0, 0
    for i in range(epochs):
        model.fit(X, y)
        if adv_valid:
            lossH.append(model.score(X_valid, y_valid))
        for j in range(inner):
            if adv_opt.lower() == 'sgd':
                X = adversary(model, X, y, enc, alpha, shared, adv, fl)
                if adv_valid:
                    X_valid = adversary(model, X_valid, y_valid, enc, alpha, shared, adv, fl)
                    X_test = adversary(model, X_test, y_test, enc, alpha, shared, adv, fl)
            elif adv_opt.lower() == 'adam':
                X, m, v = adversary_adam(model, X, y, j, m, v, enc, adv_beta, adv_eps, alpha, shared, adv, fl)
                if adv_valid:
                    X_valid, m_valid, v_valid = adversary_adam(model, X_valid, y_valid, j, m_valid, v_valid, enc,
                                                               adv_beta, adv_eps, alpha, shared, adv, fl)
                    X_test, m_test, v_test = adversary_adam(model, X_test, y_test, j, m_test, v_test, enc, adv_beta,
                                                            adv_eps, alpha, shared, adv, fl)
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

    check_folder('./data')
    np.savetxt("./data/" + head + ".csv", X, delimiter=",")
    np.savetxt("./data/" + head + "_valid.csv", X_valid, delimiter=",")
    np.savetxt("./data/" + head + "_test.csv", X_test, delimiter=",")
    np.savetxt("./data/" + head + "_y.csv", y, delimiter=",")
    np.savetxt("./data/" + head + "_y_valid.csv", y_valid, delimiter=",")
    np.savetxt("./data/" + head + "_y_test.csv", y_test, delimiter=",")

    if fl.lower() == 'none':
        X_best = np.concatenate((X_best, X_ag), axis=1)
        X_valid_best = np.concatenate((X_valid_best, X_ag_valid), axis=1)
        X_test_best = np.concatenate((X_test_best, X_ag_test), axis=1)

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


