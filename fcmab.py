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


class fcmab(object):
    def __init__(self, model, loss, opt, nc=2, n=10, epochs=10, c=.5, keep_best=True, head='', conf_matrix=False,
                 adversarial=None, step=1, plot=True):

        self.model = model  # model
        self.loss = loss  # loss
        self.opt = opt  # optimizer
        self.n = n  # number of overall iterations, sets pulled
        self.epochs = epochs  # iterations for adjusting weights
        self.nc = nc  # number of clients
        self.alpha = np.ones(nc)  # beta distribution prior
        self.beta = np.ones(nc)  # beta distribution prior
        self.theta = np.zeros(nc)  # random variable
        self.c = c  # cutoff
        self.keep_best = keep_best  # whether or not best model is kept
        self.head = head  # file label
        self.conf_matrix = conf_matrix  # whether or not a confusion matrix is generated
        self.adversarial = adversarial  # which (if any) clients are adversarial
        self.step = step  # step size of adversary
        self.plot = plot  # if validation accuracy is plotted

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
        val_acc_list = []
        for i in range(self.n):

            # open log
            old_stdout = sys.stdout  # save old output
            log_file = open('./logs/' + self.head + '.log', 'a')  # open log file
            sys.stdout = log_file  # write to log file

            # pull clients
            self.theta = np.random.beta(self.alpha, self.beta)
            print(self.theta)
            if type(self.c) == int or type(self.c) == float:
                self.model.S = self.theta > self.c
            elif self.c.lower() == 'mean':
                self.model.S = self.theta > np.mean(map)
            else:
                raise Exception('Cutoff not implemented.')
            print(self.model.S)

            # adjust weights
            print('Iter\tEpoch\tLoss')
            self.model.train()
            for epoch in range(self.epochs):

                loss_list, size, nd = [], [], []
                for _, data in enumerate(train_loader):

                    out, l, _ = self.model_loss(data)

                    self.opt.zero_grad()
                    l.backward()
                    self.opt.step()

                    loss_list.append(l.item())
                    size.append(out.shape[0])

                if self.adversarial is not None and (
                        (type(self.adversarial) == int and self.model.S[self.adversarial]) or (
                        type(self.adversarial) == list and all(self.model.S[a] for a in self.adversarial))):

                    for _, data in enumerate(train_loader):

                        _, l, _ = self.model_loss(data, adversarial=True)
                        l.backward()

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
                            x0 += self.step * x0.grad
                        if self.nc >= 2 and ((type(self.adversarial) == int and self.adversarial == 1) or
                                             (type(self.adversarial) == list and 1 in self.adversarial)):
                            x1.requires_grad = False
                            x1 += self.step * x1.grad
                        if self.nc >= 3 and ((type(self.adversarial) == int and self.adversarial == 2) or
                                             (type(self.adversarial) == list and 2 in self.adversarial)):
                            x2.requires_grad = False
                            x2 += self.step * x2.grad
                        if self.nc >= 4 and ((type(self.adversarial) == int and self.adversarial == 3) or
                                             (type(self.adversarial) == list and 3 in self.adversarial)):
                            x3.requires_grad = False
                            x3 += self.step * x3.grad

                        if self.nc == 2:
                            nd.append(utils_data.TensorDataset(x0, x1, y))
                        elif self.nc == 4:
                            nd.append(utils_data.TensorDataset(x0, x1, x2, x3, y))

                if len(nd) > 0:
                    nd = utils_data.ConcatDataset(nd)
                    train_loader = utils_data.DataLoader(nd, batch_size=train_loader.batch_size,
                                                         num_workers=train_loader.num_workers,
                                                         pin_memory=train_loader.pin_memory)
                    del nd

                # scheduler.step()
                torch.save(self.model.state_dict(), './models/' + self.head + '.pt')
                loss_avg = np.average(loss_list, weights=size)
                print("%d\t%d\t%f" % (i, epoch, loss_avg))

            self.model.eval()

            # compute validation accuracy
            val_loss, val_acc = self.loss_acc(val_loader, head=self.head + '_val')
            val_acc_list.append(val_acc)

            # keep best model
            print("current: %f, best: %f" % (val_acc, best_acc))
            if self.keep_best and val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), './models/' + self.head + '_best.pt')
                print('new high!')

            # adjust priors
            self.alpha += val_acc/100 * self.model.S
            self.beta += (1-val_acc/100) * self.model.S
            map = self.alpha/(self.alpha + self.beta)
            print(map)

            # close log
            log_file.close()  # close log file
            sys.stdout = old_stdout  # reset output

        if self.keep_best:
            self.model.load_state_dict(torch.load('./models/' + self.head + '_best.pt'))

        # open log
        old_stdout = sys.stdout  # save old output
        log_file = open('./logs/' + self.head + '.log', 'a')  # open log file
        sys.stdout = log_file  # write to log file

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
            plt.plot(val_acc_list)
            plt.title('Validation Accuracy')
            plt.savefig('./plots/' + self.head + '_valacc.png')
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
