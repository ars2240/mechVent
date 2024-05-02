from fcmab import *
from fnn import *
from floaders import *
from itertools import chain

models = ['FLRHZ']

for sh in range(1, 42, 10):
    head = 'NI+Share' + str(sh)
    if sh == 1:
        c = [[16, 19, 25, 26, 10, 11, 21, *range(111, 122)], [5, 8, 12, 24, 0, 27, 33, 34],
             [3, 9, 29, *range(41, 111), 1, 6, 30, 31], [7, 17, 32, 35, 2, 13, 18, 28],
             [14, 36, 37, *range(38, 41), 4, 15, 20, 23]]
    elif sh == 11:
        c = [[10, 11, 21, 16, 25, 26], [0, 27, 34, 5, 8, 12], [1, 6, 31, 3, 9, *range(41, 111)], [2, 13, 28, 7, 17, 35],
             [15, 20, 23, 14, 37, *range(38, 41)]]
    elif sh == 21:
        c = [[10, 11, 16, 25], [0, 5, 12, 34], [1, 3, 9, 31], [2, 7, 13, 17], [14, 15, 23, 37]]
    elif sh == 31:
        c = [[10, 16], [0, 5], [1, 3], [13, 17], [14, 15]]
    elif sh == 41:
        c = [[], [], [], [], []]
    else:
        raise Exception('Number of shared features not implemented.')
    shared = [x for x in range(0, 122) if x not in chain(*c)]
    adv = {0: [*range(len(c[0]), len(c[0])+len(shared))], 1: [*range(len(c[1]), len(c[1])+len(shared))]}
    for i in range(len(c)):
        c[i].sort()
        c[i].extend(shared)

    nf = int(sh + (41-sh)/5)
    tr_loader, val_loader, te_loader = ni_loader(batch_size=128, c=c, adv=adv, adv_valid=True)
    # tr_loader, val_loader, te_loader = adv_loader(batch_size=128, c=c, adv=adv, head='NI+10c3a_Sh' + str(sh),
    #                                               compress=True)

    for m in models:
        head2 = head + '_' + m
        if m == 'FLRSH':
            model = FLRSH(feats=c, nc=5, classes=2)
        elif m == 'FLNSH':
            model = FLNSH(feats=c, nc=5, classes=2)
        elif m == 'FLRHZ':
            model = FLRHZ(feats=c, nf=[sh, nf-sh], nc=5, classes=2)
        else:
            raise Exception('Model not found.')
        opt = torch.optim.Adam(model.parameters())
        loss = nn.CrossEntropyLoss()

        cmab = fcmab(model, loss, opt, nc=5, n=100, c='mad', head=head2 + '5c2a_RandPert_Asynch1_MAD2',
                     adv_c=[0, 1], sync=False, ucb_c=2)
        cmab.train(tr_loader, val_loader, te_loader)
