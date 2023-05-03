from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import sys
import torch
import torch.utils.data as utils_data


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


class fcmab(object):
    def __init__(self, model, loss, opt, nc=2, n=10, epochs=10, c=.5, keep_best=True, head='', conf_matrix=False,
                 adversarial=None, adv_epoch=1, adv_opt='sgd', adv_beta=(0.9, 0.999), adv_eps=1e-8, adv_step=1, plot=True,
                 m=0, ab=0, ucb_c=1, xdim=1, verbose=False):

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
        self.xdim = xdim  # dimensions of ucb x
        self.c = c  # cutoff
        self.keep_best = keep_best  # whether or not best model is kept
        self.head = head  # file label
        self.conf_matrix = conf_matrix  # whether or not a confusion matrix is generated
        self.adversarial = adversarial  # which (if any) clients are adversarial
        self.adv_epoch = adv_epoch  # number of adversarial epochs
        self.adv_opt = adv_opt  # adversarial optimizer
        self.adv_beta = adv_beta  # adversarial beta (for Adam optimizer)
        self.adv_eps = adv_eps  # adversarial epsilon (for Adam optimizer)
        self.adv_m, self.adv_v = {}, {}  # adversarial moments (for Adam optimizer)
        self.adv_step = adv_step  # step size of adversary
        self.plot = plot  # if validation accuracy is plotted
        self.m = m  # type of reward function
        self.ab = ab  # type of alpha/beta update
        self.verbose = verbose  # if extra print statements are used

    def train(self, train_loader, val_loader, test_loader):
        check_folder('./logs')
        check_folder('./models')

        # initialize log
        old_stdout = sys.stdout  # save old output
        log_file = open('./logs/' + self.head + '.log', 'w')  # open log file
        sys.stdout = log_file  # write to log file
        log_file.close()  # close log file
        sys.stdout = old_stdout  # reset output

        best_acc = 0
        map = 0.5
        tr_acc_list, val_acc_list, map_list, s_list, theta_list = [], [], [], [], []
        if 'mab' in self.c.lower():
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
        for i in range(self.n):

            # open log
            old_stdout = sys.stdout  # save old output
            log_file = open('./logs/' + self.head + '.log', 'a')  # open log file
            sys.stdout = log_file  # write to log file

            # pull clients
            self.theta = np.random.beta(self.alpha, self.beta)
            theta_list.append(self.theta)
            if self.verbose:
                print(self.theta)
            if type(self.c) == int or type(self.c) == float:
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
                    if self.verbose:
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
                    if self.verbose:
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
                self.model.S = np.array([False] * self.nc)
                self.model.S[ind[:(k+1)]] = True
                ucb_list.append(ucb)
                ucb_mean.append(ucb_m)
                ucb_std.append(ucb_s)
                self.ucb_n[k] += 1
                c = th[k]
            else:
                raise Exception('Cutoff not implemented.')

            # non-empty S
            if not any(self.model.S):
                self.model.S = self.theta == np.max(self.theta)
            if self.verbose:
                print(self.model.S)
            s_list.append(self.model.S)

            # adjust weights
            if self.verbose:
                print('Iter\tEpoch\tLoss')
            self.model.train()
            for epoch in range(self.epochs):
                for _, data in enumerate(train_loader):

                    _, l, _ = self.model_loss(data)

                    self.opt.zero_grad()
                    l.backward()
                    self.opt.step()

                if self.verbose:
                    tr_loss, _ = self.loss_acc(train_loader, head=self.head + '_tr')
                    print("%d\t%d\t%f" % (i, epoch, tr_loss))

                if self.adversarial is not None and self.adv_epoch > 1:
                    for ep in range(self.adv_epoch):
                        train_loader = self.adversary(train_loader, ep)
                        val_loader = self.adversary(val_loader, ep)
                        if self.verbose:
                            tr_loss, _ = self.loss_acc(train_loader, head=self.head + '_tr')
                            print("%s\t%d\t%f" % ('A{0}'.format(ep), epoch, tr_loss))
                elif self.adversarial is not None:
                    train_loader = self.adversary(train_loader, epoch)
                    val_loader = self.adversary(val_loader, epoch)
                    if self.verbose:
                        tr_loss, _ = self.loss_acc(train_loader, head=self.head + '_tr')
                        print("%s\t%d\t%f" % ('Adv', epoch, tr_loss))

                # scheduler.step()
                self.save()

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
                print('new high!')

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
            if self.verbose:
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

        if self.keep_best:
            self.load(head=self.head + '_best')

        # open log
        old_stdout = sys.stdout  # save old output
        log_file = open('./logs/' + self.head + '.log', 'a')  # open log file
        sys.stdout = log_file  # write to log file

        if self.verbose:
            print(self.model.S)
        print('Train\tAcc\tTest\tAcc')
        train_loss, train_acc = self.loss_acc(train_loader, head=self.head + '_train')
        test_loss, test_acc = self.loss_acc(test_loader, head=self.head + '_test')
        print("%f\t%f\t%f\t%f" % (train_loss, train_acc, test_loss, test_acc))

        # close log
        log_file.close()  # close log file
        sys.stdout = old_stdout  # reset output

        if self.plot:
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

            # subset plot
            s_list = -np.transpose(np.array(s_list).astype('int'))
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(s_list, aspect='auto', cmap=plt.cm.gray, interpolation='nearest')
            plt.title('Clients')
            plt.xlabel("Iterations")
            plt.ylabel("Client")
            ticks = np.arange(s_list.shape[0])
            ax.set_yticks(ticks)
            ax.set_yticklabels(ticks)
            plt.savefig('./plots/' + self.head + '_clients.png')
            plt.clf()
            plt.close()

            if 'mab' in self.c.lower():
                # UCB plots
                ucb_list = np.array(ucb_list)
                for i in range(ucb_list.shape[1]):
                    plt.plot(ucb_list[:, i], label=str(i+1))
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

    def model_loss(self, data, adversarial=False):
        if self.nc == 2:
            x0, x1, y = data
            x = x0, x1
        elif self.nc == 4:
            x0, x1, x2, x3, y = data
            x = x0, x1, x2, x3
        else:
            raise Exception("Number of clients not implemented.")

        if adversarial and self.adversarial is None:
            raise Exception("No adversarial client selected.")
        if adversarial and self.nc >= 1 and ((type(self.adversarial) == int and self.adversarial == 0) or
                                             (type(self.adversarial) == list and 0 in self.adversarial)):
            x0.requires_grad_()
        if adversarial and self.nc >= 2 and ((type(self.adversarial) == int and self.adversarial == 1) or
                                             (type(self.adversarial) == list and 1 in self.adversarial)):
            x1.requires_grad_()
        if adversarial and self.nc >= 3 and ((type(self.adversarial) == int and self.adversarial == 2) or
                                             (type(self.adversarial) == list and 2 in self.adversarial)):
            x2.requires_grad_()
        if adversarial and self.nc >= 4 and ((type(self.adversarial) == int and self.adversarial == 3) or
                                             (type(self.adversarial) == list and 3 in self.adversarial)):
            x3.requires_grad_()

        out = self.model(x)
        if self.model.classes == 1:
            l = self.loss(torch.squeeze(out).float(), y.float())
        else:
            l = self.loss(out, y)

        return out, l, y

    def loss_acc(self, loader, head=None):
        head = self.head if head is None else head
        loss_list, acc_list, size = [], [], []
        labels, outputs = [], []
        for _, data in enumerate(loader):

            out, l, y = self.model_loss(data)

            if self.model.classes == 1:
                predicted = (out.data > 0.5).float()
            else:
                _, predicted = torch.max(out.data, 1)

            acc = torch.mean((predicted == y).float()).item() * 100

            loss_list.append(l.item())
            acc_list.append(acc)
            size.append(out.shape[0])
            if self.conf_matrix:
                labels.extend(y)
                outputs.extend(predicted)

        if self.conf_matrix:
            conf = confusion_matrix(labels, outputs)
            np.savetxt("./logs/" + head + "_conf_matrix.csv", conf, delimiter=",")

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

                if self.nc == 2:
                    x0, x1, y = data
                elif self.nc == 4:
                    x0, x1, x2, x3, y = data
                else:
                    raise Exception("Number of clients not implemented.")

                if self.adversarial is None:
                    raise Exception("No adversarial client selected.")
                if self.nc >= 1 and ((type(self.adversarial) == int and self.adversarial == 0) or
                                     (type(self.adversarial) == list and 0 in self.adversarial)):
                    x0.requires_grad = False
                    if self.adv_opt.lower() == 'sgd':
                        x0 += step * x0.grad
                    elif self.adv_opt.lower() == 'adam':
                        self.adv_m[l][0][i] = self.adv_beta[0] * self.adv_m[l][0][i] + (1-self.adv_beta[0]) * x0.grad
                        self.adv_v[l][0][i] = self.adv_beta[1] * self.adv_v[l][0][i] + (
                                1-self.adv_beta[1]) * x0.grad**2
                        mhat = self.adv_m[l][0][i] / (1.0 - self.adv_beta[0]**(epoch+1))
                        vhat = self.adv_v[l][0][i] / (1.0 - self.adv_beta[1]**(epoch+1))
                        x0 += step * mhat / (torch.sqrt(vhat) + self.adv_eps)
                    else:
                        raise Exception('Unimplemented adversarial optimizer.')
                if self.nc >= 2 and ((type(self.adversarial) == int and self.adversarial == 1) or
                                     (type(self.adversarial) == list and 1 in self.adversarial)):
                    x1.requires_grad = False
                    if self.adv_opt.lower() == 'sgd':
                        x1 += step * x1.grad
                    elif self.adv_opt.lower() == 'adam':
                        self.adv_m[l][1][i] = self.adv_beta[0] * self.adv_m[l][1][i] + (1 - self.adv_beta[0]) * x0.grad
                        self.adv_v[l][1][i] = self.adv_beta[1] * self.adv_v[l][1][i] + (
                                    1 - self.adv_beta[1]) * x0.grad ** 2
                        mhat = self.adv_m[l][1][i] / (1.0 - self.adv_beta[0] ** (epoch + 1))
                        vhat = self.adv_v[l][1][i] / (1.0 - self.adv_beta[1] ** (epoch + 1))
                        x1 += step * mhat / (torch.sqrt(vhat) + self.adv_eps)
                    else:
                        raise Exception('Unimplemented adversarial optimizer.')
                if self.nc >= 3 and ((type(self.adversarial) == int and self.adversarial == 2) or
                                     (type(self.adversarial) == list and 2 in self.adversarial)):
                    x2.requires_grad = False
                    if self.adv_opt.lower() == 'sgd':
                        x2 += step * x2.grad
                    elif self.adv_opt.lower() == 'adam':
                        self.adv_m[l][2][i] = self.adv_beta[0] * self.adv_m[l][2][i] + (1 - self.adv_beta[0]) * x0.grad
                        self.adv_v[l][2][i] = self.adv_beta[1] * self.adv_v[l][2][i] + (
                                1 - self.adv_beta[1]) * x0.grad ** 2
                        mhat = self.adv_m[l][2][i] / (1.0 - self.adv_beta[0] ** (epoch + 1))
                        vhat = self.adv_v[l][2][i] / (1.0 - self.adv_beta[1] ** (epoch + 1))
                        x2 += step * mhat / (torch.sqrt(vhat) + self.adv_eps)
                    else:
                        raise Exception('Unimplemented adversarial optimizer.')
                if self.nc >= 4 and ((type(self.adversarial) == int and self.adversarial == 3) or
                                     (type(self.adversarial) == list and 3 in self.adversarial)):
                    x3.requires_grad = False
                    if self.adv_opt.lower() == 'sgd':
                        x3 += sstep * x3.grad
                    elif self.adv_opt.lower() == 'adam':
                        self.adv_m[l][3][i] = self.adv_beta[0] * self.adv_m[l][3][i] + (1 - self.adv_beta[0]) * x0.grad
                        self.adv_v[l][3][i] = self.adv_beta[1] * self.adv_v[l][3][i] + (
                                1 - self.adv_beta[1]) * x0.grad ** 2
                        mhat = self.adv_m[l][3][i] / (1.0 - self.adv_beta[0] ** (epoch + 1))
                        vhat = self.adv_v[l][3][i] / (1.0 - self.adv_beta[1] ** (epoch + 1))
                        x3 += step * mhat / (torch.sqrt(vhat) + self.adv_eps)
                    else:
                        raise Exception('Unimplemented adversarial optimizer.')

                if self.nc == 2:
                    nd.append(utils_data.TensorDataset(x0, x1, y))
                elif self.nc == 4:
                    nd.append(utils_data.TensorDataset(x0, x1, x2, x3, y))

            nd = utils_data.ConcatDataset(nd)
            loader = utils_data.DataLoader(nd, batch_size=loader.batch_size, num_workers=loader.num_workers,
                                           pin_memory=loader.pin_memory)

        return loader

    def save(self, head=None):
        head = self.head if head is None else head
        torch.save({'model_state_dict': self.model.state_dict(), 'S': self.model.S, 'v': self.model.v},
                   './models/' + head + '.pt')

    def load(self, head=None):
        head = self.head if head is None else head
        model = torch.load('./models/' + head + '.pt')
        self.model.load_state_dict(model['model_state_dict'])
        self.model.S = model['S']
        self.model.v = model['v']
