from fcmab import *
from fnn import *
from floaders import *
from itertools import chain

models = ['FLRSH']

for sh in range(1, 42, 20):
    head = 'NI+Share' + str(sh)
    if sh == 1:
        c = [[16, 19], [5, 24], [3, 29], [17, 32], [14, 36], [4, 15], [13, 18], [1, 30], [0, 33],
             [10, *range(111, 122)], [11, 21], [27, 34], [6, 31], [2, 28], [20, 23], [37, *range(38, 41)], [7, 35],
             [9, *range(41, 111)], [8, 12], [25, 26]]
    elif sh == 21:
        c = [[16], [5], [3], [17], [14], [15], [13], [1], [0], [10], [11], [34], [31], [2], [23], [37], [7], [9], [12],
             [25]]
    elif sh == 41:
        c = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    else:
        raise Exception('Number of shared features not implemented.')
    shared = [x for x in range(0, 122) if x not in chain(*c)]
    adv = {0: [*range(len(c[0]), len(c[0]) + len(shared))], 1: [*range(len(c[1]), len(c[1]) + len(shared))],
           2: [*range(len(c[2]), len(c[2]) + len(shared))], 3: [*range(len(c[3]), len(c[3]) + len(shared))],
           4: [*range(len(c[4]), len(c[4]) + len(shared))], 5: [*range(len(c[5]), len(c[5]) + len(shared))]}
    for i in range(len(c)):
        c[i].sort()
        c[i].extend(shared)

    nf = int(sh + (41-sh)/20)
    tr_loader, val_loader, te_loader = ni_loader(batch_size=128, c=c, adv=adv, adv_valid=True)
    # tr_loader, val_loader, te_loader = adv_loader(batch_size=128, c=c, adv=adv, head='NI+10c3a_Sh' + str(sh),
    #                                               compress=True)

    for m in models:
        head2 = head + '_' + m
        if m == 'FLRSH':
            model = FLRSH(feats=c, nc=20, classes=2)
        elif m == 'FLNSH':
            model = FLNSH(feats=c, nc=20, classes=2)
        elif m == 'FLRHZ':
            model = FLRHZ(feats=c, nf=[sh, nf-sh], nc=20, classes=2)
        else:
            raise Exception('Model not found.')
        opt = torch.optim.Adam(model.parameters())
        loss = nn.CrossEntropyLoss()

        cmab = fcmab(model, loss, opt, nc=20, n=100, c='allgood', head=head2 + '20c6a_allgood_RandPert_Reset',
                     adv_c=[*range(6)], fix_reset=True)
        cmab.train(tr_loader, val_loader, te_loader)
