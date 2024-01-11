from fcmab import *
from fnn import *
from floaders import *
from itertools import chain

for sh in range(1, 42, 10):
    head = 'NI+Share' + str(sh)
    if sh == 1:
        c = [[16, 19, 25, 26], [5, 8, 12, 24], [3, 9, 29, *range(41, 111)], [7, 17, 32, 35], [14, 36, 37, *range(38, 41)],
             [4, 15, 20, 23], [2, 13, 18, 28], [1, 6, 30, 31], [0, 27, 33, 34], [10, 11, 21, *range(111, 122)]]
    elif sh == 11:
        c = [[16, 25, 26], [5, 8, 12], [3, 9, *range(41, 111)], [7, 17, 35], [14, 37, *range(38, 41)], [15, 20, 23],
             [2, 13, 28], [1, 6, 31], [0, 27, 34], [10, 11, 21]]
    elif sh == 21:
        c = [[16, 25], [5, 12], [3, 9], [7, 17], [14, 37], [15, 23], [2, 13], [1, 31], [0, 34], [10, 11]]
    elif sh == 31:
        c = [[16], [5], [3], [17], [14], [15], [13], [1], [0], [10]]
    elif sh == 41:
        c = [[], [], [], [], [], [], [], [], [], []]
    else:
        raise Exception('Number of shared features not implemented.')
    shared = [x for x in range(0, 122) if x not in chain(*c)]
    adv = {0: [*range(len(c[0]), len(c[0])+len(shared))], 1: [*range(len(c[1]), len(c[1])+len(shared))],
           2: [*range(len(c[2]), len(c[2])+len(shared))]}
    for i in range(len(c)):
        c[i].extend(shared)

    nf = int(sh + (41-sh)/10)
    # tr_loader, val_loader, te_loader = ni_loader(batch_size=128, c=c, adv=adv, adv_valid=True)
    tr_loader, val_loader, te_loader = adv_loader(batch_size=128, c=c, adv=adv, head='NI+10c3a_Sh' + str(sh),
                                                  compress=True)
    for m in ['FLRSH', 'FLNSH']:
        head2 = head + '_' + m
        if m == 'FLRSH':
            model = FLRSH(feats=c, nc=10, classes=2)
        elif m == 'FLNSH':
            model = FLNSH(feats=c, nc=10, classes=2)
        elif m == 'FLRHZ':
            model = FLRHZ(feats=c, nf=nf, nc=10, classes=2)
        else:
            raise Exception('Model not found.')
        opt = torch.optim.Adam(model.parameters())
        loss = nn.CrossEntropyLoss()

        cmab = fcmab(model, loss, opt, nc=10, n=100, c='mablin', head=head2 + '10c3a_AdvHztl_Reset',
                     adv_c=[0, 1, 2], fix_reset=True)
        cmab.train(tr_loader, val_loader, te_loader)
