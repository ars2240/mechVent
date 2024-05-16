import copy
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import sys
import time
import torch
import torch.utils.data as utils_data
from fnn import *
from floaders import *
from itertools import chain


def check_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)


def matprod3(A, b):
    return np.matmul(np.matmul(np.transpose(b), A), b)


def dot(a, b):
    if a.ndim == 1:
        return np.dot(a, b)
    elif a.ndim == 2 and np.shape(a)[1] == 1:
        return np.matmul(np.transpose(a), b)
    elif a.ndim == 2 and np.shape(a)[0] == 1:
        return np.matmul(a, b)
    else:
        raise Exception('Bad dot product.')


def timeHMS(t, head=''):
    hrs = np.floor(t / 3600)
    t = t - hrs * 3600
    mins = np.floor(t / 60)
    secs = t - mins * 60
    print(head + 'Time elapsed: %2i hrs, %2i min, %4.2f sec' % (hrs, mins, secs))


def numberToBase(n, b):
    # https://stackoverflow.com/a/28666223
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits[::-1]


class fcmab(object):
    def __init__(self, model, loss, opt, nc=2, n=10, epochs=10, c=.5, keep_best=True, seed=1226, head='',
                 conf_matrix=False, adversarial=None, adv_epoch=1, adv_opt='sgd', adv_beta=(0.9, 0.999), adv_eps=1e-8,
                 adv_step=1, adv_c=[], plot=True, m=0, ab=0, ucb_c=1, xdim=1, verbose=1, fix_reset=False, sync=True,
                 sync_freq=1, embed_mu=1):

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.model = model  # model
        self.loss = loss  # loss
        self.opt = opt  # optimizer
        self.n = n  # number of overall iterations, sets pulled
        self.epochs = epochs  # iterations for adjusting weights
        self.nc = nc  # number of clients
        self.alpha = np.ones(nc)  # beta distribution prior
        self.beta = np.ones(nc)  # beta distribution prior
        self.theta = np.zeros(nc)  # random variable
        self.ucb_n = np.zeros(nc)  # count for UCB
        self.ucb_c = ucb_c  # confidence in UCB
        self.diff = np.zeros(nc)  # difference of models from mean
        self.xdim = xdim  # dimensions of ucb x
        self.c = c  # cutoff
        self.keep_best = keep_best  # whether best model is kept
        self.head = head  # file label
        self.conf_matrix = conf_matrix  # whether a confusion matrix is generated
        self.adversarial = adversarial  # which (if any) clients are adversarial
        self.adv_epoch = adv_epoch  # number of adversarial epochs
        self.adv_opt = adv_opt  # adversarial optimizer
        self.adv_beta = adv_beta  # adversarial beta (for Adam optimizer)
        self.adv_eps = adv_eps  # adversarial epsilon (for Adam optimizer)
        self.adv_m, self.adv_v = {}, {}  # adversarial moments (for Adam optimizer)
        self.adv_step = adv_step  # step size of adversary
        self.adv_c = adv_c  # adversarial clients for "all good" clients iteration
        self.plot = plot  # if validation accuracy is plotted
        self.m = m  # type of reward function
        self.ab = ab  # type of alpha/beta update
        self.verbose = verbose  # if extra print statements are used
        self.best_models = {}  # best models
        self.seed = seed  # random seed
        self.fix_reset = fix_reset  # true if set seed and weights at each iteration
        self.sync = sync  # true if synchronous learning, false if asynchronous (sync after each training iteration).
        self.sync_freq = sync_freq  # frequency of synchronization of models
        self.embed_mu = embed_mu  # weight of embed loss
        if fix_reset:
            self.save(head=self.head + '_init')

    def train(self, train_loader, val_loader, test_loader):

        start = time.time()

        check_folder('./logs')
        check_folder('./models')

        # initialize log
        old_stdout = sys.stdout  # save old output
        log_file = open('./logs/' + self.head + '.log', 'w')  # open log file
        sys.stdout = log_file  # write to log file
        log_file.close()  # close log file
        sys.stdout = old_stdout  # reset output

        best_acc, best_iter, all_s, good_s = 0, 'N/A', None, None
        map = 0.5
        tr_acc_list, val_acc_list, map_list, theta_list = [], [], [], []
        ucb_list, ucb_mean, ucb_std = [], [], []
        if self.c.lower() == 'mablin':
            xz = np.random.normal(size=(self.nc, self.xdim))
            A = [np.identity(self.nc + self.xdim) for _ in range(self.nc)]
            b = [np.zeros((self.nc + self.xdim, 1)) for _ in range(self.nc)]
        elif self.c.lower() == 'mablinhyb':
            xz = np.random.normal(size=(self.nc, self.xdim))
            A0 = np.identity(self.nc)
            b0 = np.zeros((self.nc, 1))
            A = [np.identity(self.xdim) for _ in range(self.nc)]
            B = [np.zeros((self.xdim, self.nc)) for _ in range(self.nc)]
            b = [np.zeros((self.xdim, 1)) for _ in range(self.nc)]
        elif self.c.lower() == 'allgood':
            good = [c for c in range(self.nc) if c not in self.adv_c]
            self.n = 2 ** len(good) - 1
            print('Setting number of iterations to {0}'.format(self.n))
        elif self.c.lower() in ['distance', 'mad']:
            self.model.S = np.array([True] * self.nc)
        good_v = [c not in self.adv_c for c in range(self.nc)]
        s_list = np.zeros((self.n, self.nc))
        for i in range(self.n):

            # open log
            old_stdout = sys.stdout  # save old output
            log_file = open('./logs/' + self.head + '.log', 'a')  # open log file
            sys.stdout = log_file  # write to log file

            # pull clients
            self.theta = np.random.beta(self.alpha, self.beta)
            theta_list.append(self.theta)
            if self.verbose >= 1:
                print(self.theta)
            if type(self.c) is int or type(self.c) is float:
                self.model.S = self.theta > self.c
                c = self.c
            elif self.c.lower() == 'mean':
                self.model.S = self.theta > np.mean(map)
                c = np.mean(map)
            elif 'mab' in self.c.lower():
                ind = np.argsort(-self.theta)
                th = self.theta[ind]
                if self.c.lower() == 'mablin':
                    x = [np.expand_dims(np.concatenate((self.theta, xz[j]), axis=None), axis=1) for j in range(self.nc)]
                    Ai = [np.linalg.inv(A[j]) for j in range(self.nc)]
                    th_hat = [np.matmul(Ai[j], b[j]) for j in range(self.nc)]
                    s = [matprod3(Ai[j], x[j]) for j in range(self.nc)]
                    ucb_m = [(dot(th_hat[j], x[j])).item() for j in range(self.nc)]
                    ucb_s = [np.sqrt(s[j]).item() for j in range(self.nc)]
                    ucb = [ucb_m[j] + self.ucb_c * ucb_s[j] for j in range(self.nc)]
                    if self.verbose >= 2:
                        print('A: {0}'.format(A))
                        print('b: {0}'.format(b))
                        print('x: {0}'.format(x))
                        print('Ai: {0}'.format(Ai))
                        print('th_hat: {0}'.format(th_hat))
                        print('s: {0}'.format(s))
                        print('ucb: {0}'.format(ucb))
                elif self.c.lower() == 'mablinhyb':
                    A0i = np.linalg.inv(A0)
                    b_hat = np.matmul(A0i, b0)
                    Ai = [np.linalg.inv(A[j]) for j in range(self.nc)]
                    th_hat = [np.matmul(Ai[j], b[j]-np.matmul(B[j], b_hat)) for j in range(self.nc)]
                    t0 = np.matmul(np.transpose(self.theta), A0i)
                    t1 = [np.matmul(np.matmul(np.matmul(t0, np.transpose(B[j])), Ai[j]), xz[j]) for j in range(self.nc)]
                    t2 = [matprod3(np.matmul(np.matmul(Ai[j], matprod3(A0i, np.transpose(B[j]))), Ai[j]), xz[j]) for j
                          in range(self.nc)]
                    s = [matprod3(A0i, self.theta) - 2 * t1[j] + matprod3(Ai[j], xz[j]) + t2[j] for j in range(self.nc)]
                    ucb_m = [(dot(self.theta, b_hat) + dot(th_hat[j], xz[j])).item() for j in range(self.nc)]
                    ucb_s = [np.sqrt(s[j]).item() for j in range(self.nc)]
                    ucb = [ucb_m[j] + self.ucb_c * ucb_s[j] for j in range(self.nc)]
                    if self.verbose >= 2:
                        print('A: {0}'.format(A))
                        print('B: {0}'.format(B))
                        print('b: {0}'.format(b))
                        print('A0: {0}'.format(A0))
                        print('b0: {0}'.format(b0))
                        print('A0i: {0}'.format(A0i))
                        print('Ai: {0}'.format(Ai))
                        print('b_hat: {0}'.format(b_hat))
                        print('th_hat: {0}'.format(th_hat))
                        print('s: {0}'.format(s))
                        print('ucb: {0}'.format(ucb))
                else:
                    ucb_m = np.cumsum(th) / np.arange(1, self.nc + 1)
                    ucb_s = np.sqrt(np.log(i) / self.ucb_n)
                    ucb = ucb_m + self.ucb_c * ucb_s if i > 0 else ucb_m
                    ucb[self.ucb_n == 0] = 1
                k = np.argmax(ucb)
                print('{0} clients selected.'.format(k+1))
                cc = [str(x) for x in ind[:(k+1)]]
                print('All Clients Chosen' if k+1 == self.nc else 'Clients Chosen: {0}'.format(', '.join(cc)))
                self.model.S = np.array([False] * self.nc)
                self.model.S[ind[:(k+1)]] = True
                ucb_list.append(ucb)
                ucb_mean.append(ucb_m)
                ucb_std.append(ucb_s)
                self.ucb_n[k] += 1
                c = th[k]
            elif self.c.lower() == 'allgood':
                self.model.S = np.array([False] * self.nc)
                bi = numberToBase(i+1, 2)
                print(bi)
                self.model.S[good[-len(bi):]] = [x == 1 for x in bi]
            elif self.c.lower() in ['distance', 'mad']:
                c = [x for x, b in enumerate(self.model.S) if b]
                m = self.diff[c]
                sd = np.median(m) if 'mad' else np.sum(np.power(m, 2))/len(c)
                if sd > 0:
                    zs = m/sd
                    if self.verbose >= 2:
                        print(zs)
                    cuts = zs < self.ucb_c
                    z2 = [np.NAN] * self.nc
                    for j in range(len(c)):
                        self.model.S[c[j]], z2[c[j]] = cuts[j], zs[j]
                    ucb_list.append(z2)
            else:
                raise Exception('Cutoff not implemented.')

            # non-empty S
            if not any(self.model.S):
                self.model.S = self.theta == np.max(self.theta)
            print(self.model.S)
            s_list[i, :] = self.model.S

            # adjust weights
            train_loader, val_loader = self.train_iter(train_loader, val_loader, i)

            val_acc, best_acc, best_iter = self.model_eval(train_loader, val_loader, tr_acc_list, val_acc_list,
                                                           best_acc, best_iter, i)

            all_s = str(self.model.S) if all(self.model.S) else all_s
            good_s = str(self.model.S) if all(self.model.S == good_v) else good_s

            # compute rewards
            if self.m == 0:
                m = val_acc/100
            elif self.m == 1:
                m = val_acc / 100 - c + .5
            else:
                raise Exception('m not implemented.')

            # adjust priors
            if self.ab == 0:
                self.alpha += m * self.model.S
                self.beta += (1-m) * self.model.S
            elif self.ab == 1:
                self.alpha += m * self.model.S + (1-m) * np.invert(self.model.S)
                self.beta += (1 - m) * self.model.S + m * np.invert(self.model.S)
            else:
                raise Exception('ab not implemented.')

            # compute MAP
            map = self.alpha/(self.alpha + self.beta)
            map_list.append(map)
            if self.verbose >= 1:
                print(map)

            if self.c.lower() == 'mablin':
                A[k] += np.outer(x[k], x[k])
                b[k] += m * x[k]
            elif self.c.lower() == 'mablinhyb':
                t = np.matmul(np.transpose(B[k]), Ai[k])
                A0 += np.matmul(t, B[k])
                b0 += np.matmul(t, b[k])
                A[k] += np.outer(xz[k], xz[k])
                B[k] += np.outer(xz[k], self.theta)
                b[k] += m * xz[k]
                t = np.matmul(np.transpose(B[k]), np.linalg.inv(A[k]))
                A0 += np.outer(self.theta, self.theta) - np.matmul(t, B[k])
                b0 += m * np.expand_dims(self.theta, axis=1) - np.matmul(t, b[k])

            # close log
            log_file.close()  # close log file
            sys.stdout = old_stdout  # reset output

        old_stdout = sys.stdout  # save old output
        log_file = open('./logs/' + self.head + '.log', 'a')  # open log file
        sys.stdout = log_file  # write to log file

        stop = time.time() - start
        timeHMS(stop)

        # close log
        log_file.close()  # close log file
        sys.stdout = old_stdout  # reset output

        self.results(train_loader, test_loader, best_acc, best_iter, all_s, good_s)

        if self.plot:
            self.plots(tr_acc_list, val_acc_list, map_list, theta_list, s_list, ucb_list, ucb_mean, ucb_std)

    def train_iter(self, train_loader, val_loader, i):
        if self.fix_reset:
            torch.manual_seed(self.seed)
            self.load(head=self.head + '_init', load_S=False)

        if self.verbose >= 1:
            head = 'Iter\tEpoch\tLoss'
            head += '\tFLoss\tEmbed\tRatio' if hasattr(self.model, 'embed_sh') else ''
            print(head)

        self.model.train()
        if not self.sync:
            c = [x for x, b in enumerate(self.model.S) if b]  # get selected clients
        for epoch in range(self.epochs):

            for j, data in enumerate(train_loader):

                if j == 0 and epoch == 0 and self.verbose >= 2:
                    X, y = data[:-1], data[-1]
                    for l in range(self.nc):
                        print('x{0}: {1}'.format(l, X[l]))
                    print('y: {0}'.format(y))

                if self.sync:
                    _, l, _ = self.model_loss(data)

                    self.opt.zero_grad()
                    l.backward()
                    self.opt.step()
                else:
                    for k in c:
                        _, l, _ = self.model_loss(data, client=k)

                        self.opt.zero_grad()
                        l.backward()
                        self.opt.step()

            if not self.sync and epoch % self.sync_freq == 0:
                # re-combine modes
                sd = self.model.state_dict()
                keys = sd.keys()
                keyVar = np.unique([k.split('.')[0] + ''.join(k.split('.')[2:]) for k in keys])

                c = [x for x, b in enumerate(self.model.S) if b]  # get selected clients
                self.diff = np.zeros(self.nc)
                for k in keyVar:
                    if 'embed' not in k:
                        keys2 = [k2 for k2 in keys if k2.split('.')[0] + ''.join(k2.split('.')[2:]) == k]
                        m = 0
                        for j in range(len(c)):
                            k2 = keys2[c[j]]
                            # print('{0} shape on client {1}: {2}'.format(k2, j, sd[k2].shape))
                            m = sd[k2] * 1 / (j + 1) + m * j / (j + 1)
                        m2 = np.median([sd[keys2[c[j]]].detach().numpy() for j in range(len(c))], axis=0) if \
                            self.c.lower() == 'mad' else m
                        m2 = torch.from_numpy(m2) if isinstance(m2, np.ndarray) else m2
                        for j in range(self.nc):
                            k2 = keys2[j]
                            self.diff[j] += np.linalg.norm(sd[k2] - m2) ** 2 if j in c else 0
                            sd[k2] = m
                self.diff = np.sqrt(self.diff)
                self.model.load_state_dict(sd)

        if self.verbose >= 1:
            if hasattr(self.model, 'embed_sh'):
                tr_loss, _, ll_loss, el_loss = self.loss_acc(train_loader, head=self.head + '_tr', split=True)
            else:
                tr_loss, _ = self.loss_acc(train_loader, head=self.head + '_tr')
            p = "%d\t%d\t%f" % (i, epoch, tr_loss)
            p += '\t%f\t%f\t%f' % (ll_loss, el_loss, el_loss/ll_loss) if hasattr(self.model, 'embed_sh') else ''
            print(p)

        if self.adversarial is not None and self.adv_epoch > 1:
            for ep in range(self.adv_epoch):
                train_loader = self.adversary(train_loader, ep)
                val_loader = self.adversary(val_loader, ep)
                if self.verbose >= 1:
                    tr_loss, _ = self.loss_acc(train_loader, head=self.head + '_tr')
                    print("A%s\t%d\t%f" % (ep, epoch, tr_loss))
        elif self.adversarial is not None:
            train_loader = self.adversary(train_loader, epoch)
            val_loader = self.adversary(val_loader, epoch)
            if self.verbose >= 1:
                tr_loss, _ = self.loss_acc(train_loader, head=self.head + '_tr')
                print("%s\t%d\t%f" % ('Adv', epoch, tr_loss))

        # scheduler.step()
        self.save()

        return train_loader, val_loader

    def model_eval(self, train_loader, val_loader, tr_acc_list, val_acc_list, best_acc, best_iter, i):
        self.model.eval()

        # compute training accuracy
        _, tr_acc = self.loss_acc(train_loader, head=self.head + '_tr')
        tr_acc_list.append(tr_acc)

        # compute validation accuracy
        val_loss, val_acc = self.loss_acc(val_loader, head=self.head + '_val')
        val_acc_list.append(val_acc)

        # keep best model
        print("current: %f, best: %f" % (val_acc, best_acc))
        if self.keep_best and val_acc > best_acc:
            best_acc = val_acc
            self.save(head=self.head + '_best')
            best_iter = i
            print('new high!')

        str_s = str(self.model.S)
        if str_s not in self.best_models.keys() or self.best_models[str_s] < val_acc:
            self.best_models[str_s] = val_acc

        return val_acc, best_acc, best_iter

    def results(self, train_loader, test_loader, best_acc, best_iter, all_s=None, good_s=None):
        # open log
        old_stdout = sys.stdout  # save old output
        log_file = open('./logs/' + self.head + '.log', 'a')  # open log file
        sys.stdout = log_file  # write to log file

        print('Results:')
        if self.keep_best:
            print('Best iteration: {0}'.format(best_iter))
            self.load(head=self.head + '_best')
        print(self.best_models)

        cc = ', '.join([str(x) for x, b in enumerate(self.model.S) if b])
        print(self.model.S)
        print('All Clients Chosen' if all(self.model.S) else 'Clients Chosen: {0}'.format(cc))
        print('Train\tAcc\tTest\tAcc')
        train_loss, train_acc = self.loss_acc(train_loader, head=self.head + '_tr')
        test_loss, test_acc = self.loss_acc(test_loader, head=self.head + '_test')
        print("%f\t%f\t%f\t%f" % (train_loss, train_acc, test_loss, test_acc))

        if self.nc == 2:
            print('Config\tTrAcc\tTeAcc\tBthAcc\tC0Acc\tC1Acc')
            bth, c0, c1 = '[ True  True]', '[ True False]', '[False  True]'
            config = 'Both' if all(self.model.S) else 'Client 0' if self.model.S[0] else 'Client 1'
            print("%s\t%f\t%f\t%f\t%f\t%f" % (config, train_acc, test_acc, self.best_models[bth], self.best_models[c0],
                                              self.best_models[c1]))
        else:
            print('Config\tTrAcc\tTeAcc\tValAcc\tAllAcc\tGoodAcc')
            config = 'All Clients' if all(self.model.S) else 'Clients {0}'.format(cc)
            all_acc = '' if all_s is None else '{0}'.format(self.best_models[all_s])
            good_acc = '' if good_s is None else '{0}'.format(self.best_models[good_s])
            print("%s\t%f\t%f\t%f\t%s\t%s" % (config, train_acc, test_acc, best_acc, all_acc, good_acc))

        # close log
        log_file.close()  # close log file
        sys.stdout = old_stdout  # reset output

    def plots(self, tr_acc_list, val_acc_list, map_list, theta_list, s_list, ucb_list, ucb_mean, ucb_std):
        check_folder('./plots')

        # accuracy plot
        plt.plot(tr_acc_list, label='Training')
        plt.plot(val_acc_list, label='Validation')
        plt.title('Accuracy')
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig('./plots/' + self.head + '_acc.png')
        plt.clf()
        plt.close()

        # map plot
        map_list = np.array(map_list)
        for i in range(map_list.shape[1]):
            plt.plot(map_list[:, i], label=str(i))
        plt.title('MAP')
        plt.xlabel("Iterations")
        plt.ylabel("MAP")
        plt.legend()
        plt.savefig('./plots/' + self.head + '_map.png')
        plt.clf()
        plt.close()

        # Theta plot
        theta_list = np.array(theta_list)
        for i in range(theta_list.shape[1]):
            plt.plot(theta_list[:, i], label=str(i))
        plt.title(r"$\theta$")
        plt.xlabel("Iterations")
        plt.ylabel(r"$\theta$")
        plt.legend()
        plt.savefig('./plots/' + self.head + '_theta.png')
        plt.clf()
        plt.close()

        # number of clients
        s_list = np.transpose(np.array(s_list).astype('int'))
        np.savetxt("./logs/" + self.head + "_clients.csv", np.transpose(s_list), delimiter=",")
        plt.plot(np.sum(s_list, axis=0))
        plt.title('Number of Clients')
        plt.xlabel("Iterations")
        plt.ylabel("Number of Clients")
        plt.ylim(0, self.nc)
        plt.savefig('./plots/' + self.head + '_numclients.png')
        plt.clf()
        plt.close()

        # subset plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(-s_list, aspect='auto', cmap=plt.cm.gray, interpolation='nearest')
        plt.title('Clients')
        plt.xlabel("Iterations")
        plt.ylabel("Client")
        ticks = np.arange(s_list.shape[0])
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticks)
        plt.savefig('./plots/' + self.head + '_clients.png')
        plt.clf()
        plt.close()

        if 'distance' in self.c.lower():
            # UCB plots
            ucb_list = np.array(ucb_list)
            for i in range(ucb_list.shape[1]):
                plt.plot(ucb_list[:, i], label=str(i + 1))
            plt.title("Distance")
            plt.xlabel("Iterations")
            plt.ylabel("Distance")
            plt.legend()
            plt.savefig('./plots/' + self.head + '_dist.png')
            plt.clf()
            plt.close()

        if 'mab' in self.c.lower():
            # UCB plots
            ucb_list = np.array(ucb_list)
            for i in range(ucb_list.shape[1]):
                plt.plot(ucb_list[:, i], label=str(i + 1))
            plt.title("Acquisition Cost")
            plt.xlabel("Iterations")
            plt.ylabel("Acquisition Cost")
            plt.legend()
            plt.savefig('./plots/' + self.head + '_ucb.png')
            plt.clf()
            plt.close()

            ucb_mean = np.array(ucb_mean)
            for i in range(ucb_mean.shape[1]):
                plt.plot(ucb_mean[:, i], label=str(i + 1))
            plt.title("Acquisition Cost Mean")
            plt.xlabel("Iterations")
            plt.ylabel("Acquisition Cost")
            plt.legend()
            plt.savefig('./plots/' + self.head + '_ucb_mean.png')
            plt.clf()
            plt.close()

            ucb_std = np.array(ucb_std)
            for i in range(ucb_std.shape[1]):
                plt.plot(ucb_std[:, i], label=str(i + 1))
            plt.title("Acquisition Cost Standard Deviation")
            plt.xlabel("Iterations")
            plt.ylabel("Acquisition Cost")
            plt.legend()
            plt.savefig('./plots/' + self.head + '_ucb_std.png')
            plt.clf()
            plt.close()

    def model_loss(self, data, client=None, adversarial=False, split=False):

        X, y = data[:-1], data[-1]

        if adversarial and self.adversarial is None:
            raise Exception("No adversarial client selected.")
        for i in range(self.nc):
            if adversarial and self.nc >= i + 1 and ((type(self.adversarial) == int and self.adversarial == i) or
                                                     (type(self.adversarial) == list and i in self.adversarial)):
                X[i].requires_grad_()

        if client is None:
            out = self.model(X)
        else:
            out = self.model(X, client)
        if self.model.classes == 1:
            l = self.loss(torch.squeeze(out).float(), y.float())
        else:
            l = self.loss(out, y)

        # add regularization for embeddings
        ll = l
        if hasattr(self.model, 'embed_sh'):
            c = [x for x, b in enumerate(self.model.S) if b]  # get selected clients

            wm, bm, el, rc = 0, 0, 0, range(len(c))
            for j in range(len(c)):
                ew, eb = self.model.embed_sh[c[j]].weight, self.model.embed_sh[c[j]].bias
                if c[j] == client:
                    rc = [j]
                if self.verbose >= 2:
                    print('Client {0} Embed Weight: {1}'.format(c[j], ew.item()))
                wm, bm = ew * 1/(j + 1) + wm * j/(j + 1), eb * 1/(j + 1) + bm * j/(j + 1)
            for j in rc:
                ew, eb = self.model.embed_sh[c[j]].weight, self.model.embed_sh[c[j]].bias
                el += torch.norm(ew - wm)
                el += torch.norm(eb - bm)
            if self.verbose >= 2:
                print('Embed Weight Mean: {0}'.format(wm.item()))
                print('Embed Loss: {0}'.format(el))
            l += self.embed_mu * el

        if split:
            return out, l, y, (ll, el)
        else:
            return out, l, y

    def loss_acc(self, loader, head=None, split=False):
        head = self.head if head is None else head
        loss_list, acc_list, size = [], [], []
        labels, outputs = [], []
        ll_list, el_list = [], []
        for _, data in enumerate(loader):

            if split:
                out, l, y, (ll, el) = self.model_loss(data, split=True)
            else:
                out, l, y = self.model_loss(data)

            if self.model.classes == 1:
                predicted = (out.data > 0.5).float()
            else:
                _, predicted = torch.max(out.data, 1)

            acc = torch.mean((predicted == y).float()).item() * 100

            loss_list.append(l.item())
            acc_list.append(acc)
            if split:
                ll_list.append(ll.item())
                el_list.append(el.item())
            size.append(out.shape[0])
            if self.conf_matrix:
                labels.extend(y)
                outputs.extend(predicted)

        if self.conf_matrix:
            conf = confusion_matrix(labels, outputs)
            np.savetxt("./logs/" + head + "_conf_matrix.csv", conf, delimiter=",")
            lp = np.concatenate((np.expand_dims(labels, axis=1), np.expand_dims(outputs, axis=1)), axis=1)
            np.savetxt("./logs/" + head + "_labels_pred.csv", lp, delimiter=",")

        if split:
            return np.average(loss_list, weights=size), np.average(acc_list, weights=size),\
                np.average(ll_list, weights=size), np.average(el_list, weights=size)
        else:
            return np.average(loss_list, weights=size), np.average(acc_list, weights=size)

    def adversary(self, loader, epoch=None):
        if self.adversarial is not None and ((type(self.adversarial) == int and self.model.S[self.adversarial]) or (
                type(self.adversarial) == list and all(self.model.S[a] for a in self.adversarial))):

            if epoch is None:
                epoch = 0
            step = self.adv_step(epoch) if callable(self.adv_step) else self.adv_step

            l = len(loader)
            if l not in self.adv_m.keys():
                self.adv_m[l] = [[0] * l] * self.nc
            if l not in self.adv_v.keys():
                self.adv_v[l] = [[0] * l] * self.nc

            nd = []
            for i, data in enumerate(loader):

                _, loss, _ = self.model_loss(data, adversarial=True)
                self.opt.zero_grad()
                loss.backward()

                X, y = data[:-1], data[-1]

                if self.adversarial is None:
                    raise Exception("No adversarial client selected.")
                for i in range(self.nc):
                    if self.nc >= i + 1 and ((type(self.adversarial) == int and self.adversarial == i) or
                                             (type(self.adversarial) == list and i in self.adversarial)):
                        X[i].requires_grad = False
                        if self.adv_opt.lower() == 'sgd':
                            X[i] += step * X[i].grad
                        elif self.adv_opt.lower() == 'adam':
                            self.adv_m[l][0][i] = self.adv_beta[0] * self.adv_m[l][0][i] + (
                                        1 - self.adv_beta[0]) * X[i].grad
                            self.adv_v[l][0][i] = self.adv_beta[1] * self.adv_v[l][0][i] + (
                                    1 - self.adv_beta[1]) * X[i].grad ** 2
                            mhat = self.adv_m[l][0][i] / (1.0 - self.adv_beta[0] ** (epoch + 1))
                            vhat = self.adv_v[l][0][i] / (1.0 - self.adv_beta[1] ** (epoch + 1))
                            X[i] += step * mhat / (torch.sqrt(vhat) + self.adv_eps)
                        else:
                            raise Exception('Unimplemented adversarial optimizer.')

                nd.append(utils_data.TensorDataset(*X, y))

            nd = utils_data.ConcatDataset(nd)
            loader = utils_data.DataLoader(nd, batch_size=loader.batch_size, num_workers=loader.num_workers,
                                           pin_memory=loader.pin_memory)

        return loader

    def save(self, head=None):
        head = self.head if head is None else head
        d = {'model_state_dict': self.model.state_dict(), 'S': self.model.S, 'optimizer': self.opt.state_dict()}
        if hasattr(self.model, 'v'):
            d['v'] = self.model.v
        torch.save(d, './models/' + head + '.pt')

    def load(self, head=None, info=False, load_S=True):
        head = self.head if head is None else head
        model = torch.load('./models/' + head + '.pt')
        if info:
            for key, value in model['model_state_dict'].items():
                np.savetxt('./models/' + head + '_' + key + '.csv', value.detach().numpy(), delimiter=",")
        self.model.load_state_dict(model['model_state_dict'])
        if load_S:
            self.model.S = model['S']
        if 'v' in model.keys():
            self.model.v = model['v']
        if 'optimizer' in model.keys():
            self.opt.load_state_dict(model['optimizer'])


def make_list(x):
    return x if type(x) is list else [x]


class main(object):
    def __init__(self, data, advs=None, c=10, advf=3, shared=None, strategy=None, model=None):
        self.data = make_list(data)  # forest, ni, or ibm
        self.advs = make_list(advs)  # RandPert or AdvHztl
        self.c = make_list(c)  # number of clients
        self.advf = make_list(advf)  # number of adversarial features
        self.shared = make_list(shared)  # number of shared features
        self.strategy = make_list(strategy)  # strategies
        self.model = make_list(model)  # models
        self.f0, self.f1, self.classes = None, None, None

    def get_shared(self, d, c):
        if d == 'forest' and c == 20:
            sh = [12]
        elif d == 'forest' and c == 10:
            sh = [*range(2, 13, 10)]
        elif d == 'forest' and c == 5:
            sh = [*range(2, 13, 5)]
        elif d == 'ni' and c == 20:
            sh = [*range(1, 42, 20)]
        elif d == 'ni' and c in [5, 10]:
            sh = [*range(1, 42, 10)]
        elif d == 'ibm' and c == 20:
            sh = [1, 81, 181, 261, 341]
        elif d == 'ibm' and c in [5, 10]:
            sh = [1, 81, 171, 251, 341]
        else:
            raise Exception('Please specify number of shared features.')
        return sh

    def get_head(self, d, sh):
        if d == 'forest':
            head = 'forest_Sh'
        elif d == 'ni':
            head = 'NI+Share'
        elif d == 'ibm':
            head = 'IBMU4_Sh'
        else:
            raise Exception('Data set not implemented.')
        head += str(sh)
        return head

    def get_f(self, d):
        if d == 'forest':
            self.f0, self.f1, self.classes = 12, 54, 7
        elif d == 'ni':
            self.f0, self.f1, self.classes = 41, 122, 2
        elif d == 'ibm':
            self.f0, self.f1, self.classes = 341, 363, 2
        else:
            raise Exception('Data set not implemented.')

    def get_c(self, d, cl, sh, advf):
        if sh == self.f0:
            c = [[] for _ in range(cl)]
            print(c)
        elif d == 'ibm' and cl == 20 and sh == 1:
            c = [[271, 287, 305, 330, 65, 264, 41, 197, 116, 4, 85, 261, 17, 141, 56, 227, 44],
                 [217, 233, 269, 270, 67, 219, 304, 218, 139, 294, 297, 31, 74, 59, 57, 320, 167],
                 [192, 201, 202, 212, *range(335, 340), 13, 25, 35, 134, 135, 136, 137, 138, 2, 71, 189, 7],
                 [149, 165, 169, 176, 38, 86, 107, 121, 129, 130, 131, 132, 133, 168, 238, 146, 318],
                 [142, 143, 144, 147, 170, 172, 174, 204, 100, 125, 126, 127, 128, 265, 148, 150, 151],
                 [115, 118, 119, 124, 206, 207, 210, 211, 284, 333, 256, 290, 98, 166, 244, 10, 242],
                 [72, 104, 110, 111, 214, 215, 216, 224, 161, 162, 163, 203, 205, 308, 28, 303, 230],
                 [34, 49, 50, 51, 331, 19, 73, 80, 156, 157, 158, 159, 160, 194, 268, 5, 63],
                 [281, 300, 311, 322, 96, 101, 178, 293, 122, 152, 153, 154, 155, 79, 257, 62, 250],
                 [145, 232, 235, 272, 36, 231, 306, 307, 199, 200, 21, 33, 120, 285, 48, 245, 263],
                 [61, 83, 89, 113, 309, 9, 55, 66, 182, 183, 184, 195, 198, 84, 299, 302, 103],
                 [312, 1, 12, 30, 191, 226, 228, 273, 90, 164, 177, 180, 181, 20, 39, 278, 29],
                 [175, 188, 209, 220, 298, 6, 82, 92, 43, 326, 280, 97, 0, 60, 254, 251, 23],
                 [327, 52, 171, 173, 225, 319, 40, 47, 88, 112, 258, 314, 332, 53, 255, 296, 282],
                 [213, 247, 317, 325, 75, 77, 78, 117, 275, 313, 292, 3, 76, 8, 241, 277, 24],
                 [108, 109, 179, 185, 248, 267, 123, 208, 187, 222, 236, 26, 37, 27, 243, 234, 276],
                 [11, 64, 81, 94, 239, 295, 321, 329, 323, 87, 246, 14, 186, 283, 105, 229, 316],
                 [58, 259, 310, 315, 334, *range(340, 345), 70, 102, 95, 252, 42, 240, 106, 93, 140, 262, 260],
                 [324, *range(352, 358), 45, 46, 289, 291, 328, *range(345, 348), 221, 193, 16, 274, 22, 279, 54,
                  253, 18],
                 [99, 114, 196, 286, *range(358, 363), *range(348, 352), 32, 69, 190, 15, 223, 249, 91, 237, 288,
                  301, 68]]
        elif d == 'ibm' and cl == 20 and sh == 81:
            c = [[271, 287, 305, 330, 65, 264, 41, 197, 116, 4, 85, 261, 17],
                 [217, 233, 269, 270, 67, 219, 304, 218, 139, 294, 297, 31, 74],
                 [192, 201, 202, 212, *range(335, 340), 13, 25, 35, 134, 135, 136, 137, 138],
                 [149, 165, 169, 176, 38, 86, 107, 121, 129, 130, 131, 132, 133],
                 [142, 143, 144, 147, 170, 172, 174, 204, 100, 125, 126, 127, 128],
                 [115, 118, 119, 124, 206, 207, 210, 211, 284, 333, 256, 290, 98],
                 [72, 104, 110, 111, 214, 215, 216, 224, 161, 162, 163, 203, 205],
                 [34, 49, 50, 51, 331, 19, 73, 80, 156, 157, 158, 159, 160],
                 [281, 300, 311, 322, 96, 101, 178, 293, 122, 152, 153, 154, 155],
                 [145, 232, 235, 272, 36, 231, 306, 307, 199, 200, 21, 33, 120],
                 [61, 83, 89, 113, 309, 9, 55, 66, 182, 183, 184, 195, 198],
                 [312, 1, 12, 30, 191, 226, 228, 273, 90, 164, 177, 180, 181],
                 [175, 188, 209, 220, 298, 6, 82, 92, 43, 326, 280, 97, 0],
                 [327, 52, 171, 173, 225, 319, 40, 47, 88, 112, 258, 314, 332],
                 [213, 247, 317, 325, 75, 77, 78, 117, 275, 313, 292, 3, 76],
                 [108, 109, 179, 185, 248, 267, 123, 208, 187, 222, 236, 26, 37],
                 [11, 64, 81, 94, 239, 295, 321, 329, 323, 87, 246, 14, 186],
                 [58, 259, 310, 315, 334, *range(340, 345), 70, 102, 95, 252, 42, 240, 106],
                 [324, *range(352, 358), 45, 46, 289, 291, 328, *range(345, 348), 221, 193, 16, 274, 22],
                 [99, 114, 196, 286, *range(358, 363), *range(348, 352), 32, 69, 190, 15, 223, 249, 91]]
        elif d == 'ibm' and cl == 20 and sh == 181:
            c = [[271, 287, 305, 330, 65, 264, 41, 197], [217, 233, 269, 270, 67, 219, 304, 218],
                 [192, 201, 202, 212, *range(335, 340), 13, 25, 35], [149, 165, 169, 176, 38, 86, 107, 121],
                 [142, 143, 144, 147, 170, 172, 174, 204], [115, 118, 119, 124, 206, 207, 210, 211],
                 [72, 104, 110, 111, 214, 215, 216, 224], [34, 49, 50, 51, 331, 19, 73, 80],
                 [281, 300, 311, 322, 96, 101, 178, 293], [145, 232, 235, 272, 36, 231, 306, 307],
                 [61, 83, 89, 113, 309, 9, 55, 66], [312, 1, 12, 30, 191, 226, 228, 273],
                 [175, 188, 209, 220, 298, 6, 82, 92], [327, 52, 171, 173, 225, 319, 40, 47],
                 [213, 247, 317, 325, 75, 77, 78, 117], [108, 109, 179, 185, 248, 267, 123, 208],
                 [11, 64, 81, 94, 239, 295, 321, 329], [58, 259, 310, 315, 334, *range(340, 345), 70, 102],
                 [324, *range(352, 358), 45, 46, 289, 291, 328, *range(345, 348)],
                 [99, 114, 196, 286, *range(358, 363), *range(348, 352), 32, 69]]
        elif d == 'ibm' and cl == 20 and sh == 261:
            c = [[271, 287, 305, 330], [217, 233, 269, 270], [192, 201, 202, 212], [149, 165, 169, 176],
                 [142, 143, 144, 147], [115, 118, 119, 124], [72, 104, 110, 111], [34, 49, 50, 51],
                 [281, 300, 311, 322],
                 [145, 232, 235, 272], [61, 83, 89, 113], [312, 1, 12, 30], [175, 188, 209, 220],
                 [327, 52, 171, 173],
                 [213, 247, 317, 325], [108, 109, 179, 185], [11, 64, 81, 94], [58, 259, 310, 315],
                 [324, *range(352, 358), 45, 46], [99, 114, 196, 286]]
        elif d == 'ibm' and cl == 10 and sh == 1:
            c = [
                [141, 56, 227, 44, 59, 57, 320, 167, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 294, 297, 31, 74,
                 116, 4, 85, 212, 217, 233, 269, 270, 271, 287, 305, 330],
                [2, 71, 189, 7, 168, 238, 146, 318, 256, 290, 98, 100, 125, 126, 127, 128, 129, 261, 17, 65, 264, 41,
                 197,
                 67, 219, 144, 147, 149, 165, 169, 176, 192, 201, 202],
                [265, 148, 150, 151, 166, 244, 10, 242, 159, 160, 161, 162, 163, 203, 205, 284, 333, 304, 218,
                 *range(335, 340), 13, 25, 35, 38, 86, 104, 110, 111, 115, 118, 119, 124, 142, 143],
                [308, 28, 303, 230, 194, 268, 5, 63, 120, 122, 152, 153, 154, 155, 156, 157, 158, 107, 121, 170, 172,
                 174,
                 204, 206, 207, 281, 300, 311, 322, 34, 49, 50, 51, 72],
                [79, 257, 62, 250, 285, 48, 245, 263, 182, 183, 184, 195, 198, 199, 200, 21, 33, 210, 211, 214, 215,
                 216,
                 224, 331, 19, 30, 61, 83, 89, 113, 145, 232, 235, 272],
                [84, 299, 302, 103, 20, 39, 278, 29, 326, 280, 97, 0, 90, 164, 177, 180, 181, 73, 80, 96, 101, 178, 293,
                 36, 231, 171, 173, 175, 188, 209, 220, 312, 1, 12],
                [60, 254, 251, 23, 53, 255, 296, 282, 292, 3, 76, 88, 112, 258, 314, 332, 43, 306, 307, 309, 9, 55, 66,
                 191, 226, 109, 179, 185, 213, 247, 317, 325, 327, 52],
                [8, 241, 277, 24, 27, 243, 234, 276, 14, 186, 187, 222, 236, 26, 37, 275, 313, 228, 273, 298, 6, 82, 92,
                 225, 319, 58, 259, 310, 315, 11, 64, 81, 94, 108],
                [283, 105, 229, 316, 93, 140, 262, 260, 22, 95, 252, 42, 240, 106, 323, 87, 246, 40, 47, 75, 77, 78,
                 117,
                 248, 267, 69, 99, 114, 196, 286, 324, *range(352, 358), 45, 46],
                [279, 54, 253, 18, 237, 288, 301, 68, 190, 15, 223, 249, 91, 221, 193, 16, 274, 123, 208, 239, 295, 321,
                 329, 334, *range(340, 345), 70, 102, 289, 291, 328, *range(345, 348), *range(358, 363),
                 *range(348, 352),
                 32]]
        elif d == 'ibm' and cl == 10 and sh == 81:
            c = [[130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 294, 297, 31, 74, 116, 4, 85, 212, 217, 233, 269,
                  270,
                  271, 287, 305, 330],
                 [256, 290, 98, 100, 125, 126, 127, 128, 129, 261, 17, 65, 264, 41, 197, 67, 219, 144, 147, 149, 165,
                  169,
                  176, 192, 201, 202],
                 [159, 160, 161, 162, 163, 203, 205, 284, 333, 304, 218, *range(335, 340), 13, 25, 35, 38, 86, 104, 110,
                  111, 115, 118, 119, 124, 142, 143],
                 [120, 122, 152, 153, 154, 155, 156, 157, 158, 107, 121, 170, 172, 174, 204, 206, 207, 281, 300, 311,
                  322,
                  34, 49, 50, 51, 72],
                 [182, 183, 184, 195, 198, 199, 200, 21, 33, 210, 211, 214, 215, 216, 224, 331, 19, 30, 61, 83, 89, 113,
                  145, 232, 235, 272],
                 [326, 280, 97, 0, 90, 164, 177, 180, 181, 73, 80, 96, 101, 178, 293, 36, 231, 171, 173, 175, 188, 209,
                  220, 312, 1, 12],
                 [292, 3, 76, 88, 112, 258, 314, 332, 43, 306, 307, 309, 9, 55, 66, 191, 226, 109, 179, 185, 213, 247,
                  317,
                  325, 327, 52],
                 [14, 186, 187, 222, 236, 26, 37, 275, 313, 228, 273, 298, 6, 82, 92, 225, 319, 58, 259, 310, 315, 11,
                  64,
                  81, 94, 108],
                 [22, 95, 252, 42, 240, 106, 323, 87, 246, 40, 47, 75, 77, 78, 117, 248, 267, 69, 99, 114, 196, 286,
                  324,
                  *range(352, 358), 45, 46],
                 [190, 15, 223, 249, 91, 221, 193, 16, 274, 123, 208, 239, 295, 321, 329, 334, *range(340, 345), 70,
                  102,
                  289, 291, 328, *range(345, 348), *range(358, 363), *range(348, 352), 32]]
        elif d == 'ibm' and cl == 10 and sh == 171:
            c = [[139, 294, 297, 31, 74, 116, 4, 85, 212, 217, 233, 269, 270, 271, 287, 305, 330],
                 [261, 17, 65, 264, 41, 197, 67, 219, 144, 147, 149, 165, 169, 176, 192, 201, 202],
                 [304, 218, *range(335, 340), 13, 25, 35, 38, 86, 104, 110, 111, 115, 118, 119, 124, 142, 143],
                 [107, 121, 170, 172, 174, 204, 206, 207, 281, 300, 311, 322, 34, 49, 50, 51, 72],
                 [210, 211, 214, 215, 216, 224, 331, 19, 30, 61, 83, 89, 113, 145, 232, 235, 272],
                 [73, 80, 96, 101, 178, 293, 36, 231, 171, 173, 175, 188, 209, 220, 312, 1, 12],
                 [306, 307, 309, 9, 55, 66, 191, 226, 109, 179, 185, 213, 247, 317, 325, 327, 52],
                 [228, 273, 298, 6, 82, 92, 225, 319, 58, 259, 310, 315, 11, 64, 81, 94, 108],
                 [40, 47, 75, 77, 78, 117, 248, 267, 69, 99, 114, 196, 286, 324, *range(352, 358), 45, 46],
                 [123, 208, 239, 295, 321, 329, 334, *range(340, 345), 70, 102, 289, 291, 328, *range(345, 348),
                  *range(358, 363), *range(348, 352), 32]]
        elif d == 'ibm' and cl == 10 and sh == 251:
            c = [[212, 217, 233, 269, 270, 271, 287, 305, 330], [144, 147, 149, 165, 169, 176, 192, 201, 202],
                 [104, 110, 111, 115, 118, 119, 124, 142, 143], [281, 300, 311, 322, 34, 49, 50, 51, 72],
                 [30, 61, 83, 89, 113, 145, 232, 235, 272], [171, 173, 175, 188, 209, 220, 312, 1, 12],
                 [109, 179, 185, 213, 247, 317, 325, 327, 52], [58, 259, 310, 315, 11, 64, 81, 94, 108],
                 [69, 99, 114, 196, 286, 324, *range(352, 358), 45, 46],
                 [70, 102, 289, 291, 328, *range(345, 348), *range(358, 363), *range(348, 352), 32]]
        elif d == 'ibm' and cl == 5 and sh == 1:
            c = [
                [141, 56, 227, 44, 59, 57, 320, 167, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 294, 297, 31, 74,
                 116, 4, 85, 212, 217, 233, 269, 270, 271, 287, 305, 330, 279, 54, 253, 18, 237, 288, 301, 68, 190, 15,
                 223, 249, 91, 221, 193, 16, 274, 123, 208, 239, 295, 321, 329, 334, *range(340, 345), 70, 102, 289,
                 291,
                 328, *range(345, 348), *range(358, 363), *range(348, 352), 32],
                [2, 71, 189, 7, 168, 238, 146, 318, 256, 290, 98, 100, 125, 126, 127, 128, 129, 261, 17, 65, 264, 41,
                 197,
                 67, 219, 144, 147, 149, 165, 169, 176, 192, 201, 202, 283, 105, 229, 316, 93, 140, 262, 260, 22, 95,
                 252,
                 42, 240, 106, 323, 87, 246, 40, 47, 75, 77, 78, 117, 248, 267, 69, 99, 114, 196, 286, 324,
                 *range(352, 358), 45, 46],
                [265, 148, 150, 151, 166, 244, 10, 242, 159, 160, 161, 162, 163, 203, 205, 284, 333, 304, 218,
                 *range(335, 340), 13, 25, 35, 38, 86, 104, 110, 111, 115, 118, 119, 124, 142, 143, 8, 241, 277, 24, 27,
                 243, 234, 276, 14, 186, 187, 222, 236, 26, 37, 275, 313, 228, 273, 298, 6, 82, 92, 225, 319, 58, 259,
                 310,
                 315, 11, 64, 81, 94, 108],
                [308, 28, 303, 230, 194, 268, 5, 63, 120, 122, 152, 153, 154, 155, 156, 157, 158, 107, 121, 170, 172,
                 174,
                 204, 206, 207, 281, 300, 311, 322, 34, 49, 50, 51, 72, 60, 254, 251, 23, 53, 255, 296, 282, 292, 3, 76,
                 88, 112, 258, 314, 332, 43, 306, 307, 309, 9, 55, 66, 191, 226, 109, 179, 185, 213, 247, 317, 325, 327,
                 52],
                [79, 257, 62, 250, 285, 48, 245, 263, 182, 183, 184, 195, 198, 199, 200, 21, 33, 210, 211, 214, 215,
                 216,
                 224, 331, 19, 30, 61, 83, 89, 113, 145, 232, 235, 272, 84, 299, 302, 103, 20, 39, 278, 29, 326, 280,
                 97,
                 0, 90, 164, 177, 180, 181, 73, 80, 96, 101, 178, 293, 36, 231, 171, 173, 175, 188, 209, 220, 312, 1,
                 12]]
        elif d == 'ibm' and cl == 5 and sh == 81:
            c = [[130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 294, 297, 31, 74, 116, 4, 85, 212, 217, 233, 269,
                  270,
                  271, 287, 305, 330, 190, 15, 223, 249, 91, 221, 193, 16, 274, 123, 208, 239, 295, 321, 329, 334,
                  *range(340, 345), 70, 102, 289, 291, 328, *range(345, 348), *range(358, 363), *range(348, 352), 32],
                 [256, 290, 98, 100, 125, 126, 127, 128, 129, 261, 17, 65, 264, 41, 197, 67, 219, 144, 147, 149, 165,
                  169,
                  176, 192, 201, 202, 22, 95, 252, 42, 240, 106, 323, 87, 246, 40, 47, 75, 77, 78, 117, 248, 267, 69,
                  99,
                  114, 196, 286, 324, *range(352, 358), 45, 46],
                 [159, 160, 161, 162, 163, 203, 205, 284, 333, 304, 218, *range(335, 340), 13, 25, 35, 38, 86, 104, 110,
                  111, 115, 118, 119, 124, 142, 143, 14, 186, 187, 222, 236, 26, 37, 275, 313, 228, 273, 298, 6, 82, 92,
                  225, 319, 58, 259, 310, 315, 11, 64, 81, 94, 108],
                 [120, 122, 152, 153, 154, 155, 156, 157, 158, 107, 121, 170, 172, 174, 204, 206, 207, 281, 300, 311,
                  322,
                  34, 49, 50, 51, 72, 292, 3, 76, 88, 112, 258, 314, 332, 43, 306, 307, 309, 9, 55, 66, 191, 226, 109,
                  179,
                  185, 213, 247, 317, 325, 327, 52],
                 [182, 183, 184, 195, 198, 199, 200, 21, 33, 210, 211, 214, 215, 216, 224, 331, 19, 30, 61, 83, 89, 113,
                  145, 232, 235, 272, 326, 280, 97, 0, 90, 164, 177, 180, 181, 73, 80, 96, 101, 178, 293, 36, 231, 171,
                  173,
                  175, 188, 209, 220, 312, 1, 12]]
        elif d == 'ibm' and cl == 5 and sh == 171:
            c = [[139, 294, 297, 31, 74, 116, 4, 85, 212, 217, 233, 269, 270, 271, 287, 305, 330, 123, 208, 239, 295,
                  321,
                  329, 334, *range(340, 345), 70, 102, 289, 291, 328, *range(345, 348), *range(358, 363),
                  *range(348, 352),
                  32],
                 [261, 17, 65, 264, 41, 197, 67, 219, 144, 147, 149, 165, 169, 176, 192, 201, 202, 40, 47, 75, 77, 78,
                  117,
                  248, 267, 69, 99, 114, 196, 286, 324, *range(352, 358), 45, 46],
                 [304, 218, *range(335, 340), 13, 25, 35, 38, 86, 104, 110, 111, 115, 118, 119, 124, 142, 143, 228, 273,
                  298, 6, 82, 92, 225, 319, 58, 259, 310, 315, 11, 64, 81, 94, 108],
                 [107, 121, 170, 172, 174, 204, 206, 207, 281, 300, 311, 322, 34, 49, 50, 51, 72, 306, 307, 309, 9, 55,
                  66,
                  191, 226, 109, 179, 185, 213, 247, 317, 325, 327, 52],
                 [210, 211, 214, 215, 216, 224, 331, 19, 30, 61, 83, 89, 113, 145, 232, 235, 272, 73, 80, 96, 101, 178,
                  293,
                  36, 231, 171, 173, 175, 188, 209, 220, 312, 1, 12]]
        elif d == 'ibm' and cl == 5 and sh == 251:
            c = [[212, 217, 233, 269, 270, 271, 287, 305, 330, 70, 102, 289, 291, 328, *range(345, 348),
                  *range(358, 363),
                  *range(348, 352), 32],
                 [144, 147, 149, 165, 169, 176, 192, 201, 202, 69, 99, 114, 196, 286, 324, *range(352, 358), 45, 46],
                 [104, 110, 111, 115, 118, 119, 124, 142, 143, 58, 259, 310, 315, 11, 64, 81, 94, 108],
                 [281, 300, 311, 322, 34, 49, 50, 51, 72, 109, 179, 185, 213, 247, 317, 325, 327, 52],
                 [30, 61, 83, 89, 113, 145, 232, 235, 272, 171, 173, 175, 188, 209, 220, 312, 1, 12]]
        elif d == 'ni' and cl == 20 and sh == 1:
            c = [[16, 19], [5, 24], [3, 29], [17, 32], [14, 36], [4, 15], [13, 18], [1, 30], [0, 33],
                 [10, *range(111, 122)], [11, 21], [27, 34], [6, 31], [2, 28], [20, 23], [37, *range(38, 41)], [7, 35],
                 [9, *range(41, 111)], [8, 12], [25, 26]]
        elif d == 'ni' and cl == 20 and sh == 21:
            c = [[16], [5], [3], [17], [14], [15], [13], [1], [0], [10], [11], [34], [31], [2], [23], [37], [7], [9], [12],
                 [25]]
        elif d == 'ni' and cl == 10 and sh == 1:
            c = [[16, 19, 25, 26], [5, 8, 12, 24], [3, 9, 29, *range(41, 111)], [7, 17, 32, 35], [14, 36, 37, *range(38, 41)],
                 [4, 15, 20, 23], [2, 13, 18, 28], [1, 6, 30, 31], [0, 27, 33, 34], [10, 11, 21, *range(111, 122)]]
        elif d == 'ni' and cl == 10 and sh == 11:
            c = [[16, 25, 26], [5, 8, 12], [3, 9, *range(41, 111)], [7, 17, 35], [14, 37, *range(38, 41)], [15, 20, 23],
                 [2, 13, 28], [1, 6, 31], [0, 27, 34], [10, 11, 21]]
        elif d == 'ni' and cl == 10 and sh == 21:
            c = [[16, 25], [5, 12], [3, 9], [7, 17], [14, 37], [15, 23], [2, 13], [1, 31], [0, 34], [10, 11]]
        elif d == 'ni' and cl == 10 and sh == 31:
            c = [[16], [5], [3], [17], [14], [15], [13], [1], [0], [10]]
        elif d == 'ni' and cl == 5 and sh == 1:
            c = [[16, 19, 25, 26, 10, 11, 21, *range(111, 122)], [5, 8, 12, 24, 0, 27, 33, 34],
                 [3, 9, 29, *range(41, 111), 1, 6, 30, 31], [7, 17, 32, 35, 2, 13, 18, 28],
                 [14, 36, 37, *range(38, 41), 4, 15, 20, 23]]
        elif d == 'ni' and cl == 5 and sh == 11:
            c = [[10, 11, 21, 16, 25, 26], [0, 27, 34, 5, 8, 12], [1, 6, 31, 3, 9, *range(41, 111)],
                 [2, 13, 28, 7, 17, 35],
                 [15, 20, 23, 14, 37, *range(38, 41)]]
        elif d == 'ni' and cl == 5 and sh == 21:
            c = [[10, 11, 16, 25], [0, 5, 12, 34], [1, 3, 9, 31], [2, 7, 13, 17], [14, 15, 23, 37]]
        elif d == 'ni' and cl == 5 and sh == 31:
            c = [[10, 16], [0, 5], [1, 3], [13, 17], [14, 15]]
        elif d == 'forest' and cl == 10 and sh == 2:
            c = [[6], [9], [4], [8], [1], [5], [*range(14, 54)], [2], [*range(10, 14)], [3]]
        elif d == 'forest' and cl == 5 and sh == 2:
            c = [[3, 6], [9, *range(10, 14)], [2, 4], [8, *range(14, 54)], [1, 5]]
        elif d == 'forest' and cl == 5 and sh == 7:
            c = [[6], [9], [4], [8], [1]]
        else:
            raise Exception('Number of shared features not implemented.')
        shared = [x for x in range(self.f1) if x not in chain(*c)]
        adv = {i: [*range(len(c[i]), len(c[i]) + len(shared))] for i in range(advf)}
        for i in range(len(c)):
            c[i].sort()
            c[i].extend(shared)
        return c, adv

    def get_loaders(self, d, advs, c, adv, sh):
        if d == 'forest' and advs == 'RandPert':
            tr_loader, val_loader, te_loader = forest_loader(batch_size=128, c=c, adv=adv, adv_valid=True)
        elif d == 'ni' and advs == 'RandPert':
            tr_loader, val_loader, te_loader = ni_loader(batch_size=128, c=c, adv=adv, adv_valid=True)
        elif d == 'ibm' and advs == 'RandPert':
            tr_loader, val_loader, te_loader = ibm_loader(batch_size=128, c=c, adv=adv, adv_valid=True, undersample=4)
        elif advs == 'AdvHztl':
            head = 'IMBU4' if d == 'ibm' else 'NI+' if d == 'ni' else 'Forest' if d == 'forest' else ''
            head += '{0}c{1}a_Sh{2}'.format(len(c), len(adv), sh)
            tr_loader, val_loader, te_loader = adv_loader(batch_size=128, c=c, adv=adv, head=head, compress=True)
        else:
            raise Exception('Data source not implemented.')
        return tr_loader, val_loader, te_loader

    def get_strategies(self):
        return ['allgood', 'mab', 'mad']

    def get_models(self, strat):
        return ['FLRHZ'] if strat in ['mad'] else ['FLRSH']

    def get_model(self, m, c, cl, sh):
        nf = int(sh + (self.f1 - sh) / cl)
        if m == 'FLRSH':
            model = FLRSH(feats=c, nc=cl, classes=self.classes)
        elif m == 'FLNSH':
            model = FLNSH(feats=c, nc=cl, classes=self.classes)
        elif m == 'FLRHZ':
            model = FLRHZ(feats=c, nf=[sh, nf - sh], nc=cl, classes=self.classes)
        else:
            raise Exception('Model not found.')
        return model

    def get_opt(self, m, d):
        return torch.optim.Adam(m.parameters(), weight_decay=.01) if d == 'ibm' else torch.optim.Adam(m.parameters())

    def get_advf(self, cl):
        if cl == 5:
            advf = [2, 3]
        elif cl == 10:
            advf = [3, 5, 7]
        elif cl == 20:
            advf = [10, 15]
        else:
            raise Exception('Number of clients not implemented. Please specify number of adversarial features.')
        return advf

    def get_advs(self):
        return ['RandPert']

    def run(self):
        for d in self.data:
            d = d.lower()
            self.get_f(d)
            for cl in self.c:
                advfs = self.get_advf(d, cl) if self.advf[0] is None else self.advf
                for advf in advfs:
                    advss = self.get_advs() if self.advs[0] is None else self.advs
                    for advs in advss:
                        strategy = self.get_strategies() if self.strategy[0] is None else self.strategy
                        for strat in strategy:
                            shared = self.get_shared(d, cl) if self.shared[0] is None else self.shared
                            for sh in shared:
                                head = self.get_head(d, sh)
                                c, adv = self.get_c(d, cl, sh, advf)
                                tr_loader, val_loader, te_loader = self.get_loaders(d, advs, c, adv, sh)
                                models = self.get_models(strat) if self.model[0] is None else self.model
                                for m in models:
                                    head2 = head + '_' + m
                                    model = self.get_model(m, c, cl, sh)
                                    opt = self.get_opt(model, d)
                                    loss = nn.CrossEntropyLoss()
                                    if strat == 'allgood':
                                        tail = '{0}c{1}a_allgood_{2}_Reset'.format(cl, advf, advs)
                                        cmab = fcmab(model, loss, opt, nc=cl, n=100, c='allgood', head=head2 + tail,
                                                     adv_c=[*range(advf)], fix_reset=True)
                                    elif strat == 'mab':
                                        tail = '{0}c{1}a_{2}_Reset'.format(cl, advf, advs)
                                        cmab = fcmab(model, loss, opt, nc=cl, n=100, c='mablin', head=head2 + tail,
                                                     adv_c=[*range(advf)], fix_reset=True)
                                    elif strat == 'mad':
                                        tail = '{0}c{1}a_{2}_Asynch1_MAD2'.format(cl, advf, advs)
                                        cmab = fcmab(model, loss, opt, nc=cl, n=100, c='mad', head=head2 + tail,
                                                     adv_c=[*range(advf)], sync=False, ucb_c=2)
                                    else:
                                        raise Exception('Config not implemented.')
                                    print(head2+tail)
                                    cmab.train(tr_loader, val_loader, te_loader)
