import numpy as np
from ni_data import *
from sklearn.metrics import confusion_matrix

batch_size = 128
epochs = 10
shared = 'flag'
# shared = ['dst_host_same_src_port_rate', 'service', 'dst_host_srv_count']
head = 'fl11_top2_7feat'
keep_best = True
conf_matrix = False
reduced = True
classes = 2
swaps = [0.5, 1]
hosts = list(range(4))

if type(shared) == str:
    head += '_' + shared
elif type(shared) == list:
    head += '_' + '_'.join(shared)
else:
    raise Exception('Improper shared type.')

loss = nn.CrossEntropyLoss()


def model_loss(data):
    x0, x1, x2, x3, y = data
    x = x0, x1, x2, x3

    out = model(x)
    l = loss(out, y)

    return out, l, y


def loss_acc(loader, conf_matrix=False, head=head):
    loss_list, acc_list, size = [], [], []
    labels, outputs = [], []
    for _, data in enumerate(loader):
        out, l, y = model_loss(data)

        _, predicted = torch.max(out.data, 1)

        acc = torch.mean((predicted == y).float()).item() * 100

        loss_list.append(l.item())
        acc_list.append(acc)
        size.append(out.shape[0])
        if conf_matrix:
            labels.extend(y)
            outputs.extend(predicted)

    if conf_matrix:
        conf = confusion_matrix(labels, outputs)
        np.savetxt("./logs/" + head + "_conf_matrix.csv", conf, delimiter=",")

    return np.average(loss_list, weights=size), np.average(acc_list, weights=size)


check_folder('./logs')
for swap in swaps:
    head2 = head + '_swap' + str(swap)
    for cur_host in hosts:

        # """
        old_stdout = sys.stdout  # save old output
        if cur_host == 0:
            log_file = open('./logs/' + head2 + '.log', 'w')  # open log file
        else:
            log_file = open('./logs/' + head2 + '.log', 'a')  # open log file

        sys.stdout = log_file  # write to log file
        # """

        print(cur_host)

        train_loader, test_loader, model = ni_loader(batch_size=batch_size, cur_host=cur_host, shared=shared, swap=swap,
                                                     classes=classes, reduced=reduced)

        def alpha(k):
            return 1/(1+k)

        check_folder('./models')

        opt = torch.optim.Adam(model.parameters())
        # opt = torch.optim.SGD(model.parameters(), lr=.5)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=alpha)

        # Training
        print('Epoch\tLoss')
        model.train()
        best_loss = np.Inf
        for epoch in range(epochs):

            loss_list, size = [], []
            for _, data in enumerate(train_loader):

                out, l, _ = model_loss(data)

                opt.zero_grad()
                l.backward()
                opt.step()

                loss_list.append(l.item())
                size.append(out.shape[0])

            # scheduler.step()
            torch.save(model.state_dict(), './models/' + head2 + '_' + str(cur_host) + '.pt')
            loss_avg = np.average(loss_list, weights=size)
            if keep_best and loss_avg < best_loss:
                best_loss = loss_avg
                torch.save(model.state_dict(), './models/' + head2 + '_' + str(cur_host) + '_best.pt')
            if not keep_best:
                best_loss = loss_avg
            print("%d\t%f" % (epoch, loss_avg))

        if keep_best:
            model.load_state_dict(torch.load('./models/' + head2 + '_' + str(cur_host) + '_best.pt'))

        print('Host\tTrain\tAcc\tTest\tAcc')
        train_loss, train_acc = loss_acc(train_loader, conf_matrix=conf_matrix, head=head2 + '_train')
        test_loss, test_acc = loss_acc(test_loader, conf_matrix=conf_matrix, head=head2 + '_test')
        print("%d\t%f\t%f\t%f\t%f" % (cur_host, train_loss, train_acc, test_loss, test_acc))

        # """
        log_file.close()  # close log file
        sys.stdout = old_stdout  # reset output
        # """
