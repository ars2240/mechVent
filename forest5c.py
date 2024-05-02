from fcmab import *
from fnn import *
from floaders import *
from itertools import chain

models = ['FLRHZ']

for sh in range(2, 13, 5):
    if sh == 2:
        c = [[3, 6], [9, *range(10, 14)], [2, 4], [8, *range(14, 54)], [1, 5]]
    elif sh == 7:
        c = [[6], [9], [4], [8], [1]]
    elif sh == 12:
        c = [[], [], [], [], []]
    else:
        raise Exception('Number of shared features not implemented.')
    shared = [x for x in range(0, 54) if x not in chain(*c)]
    adv = {0: [*range(len(c[0]), len(c[0]) + len(shared))], 1: [*range(len(c[1]), len(c[1]) + len(shared))]}
    for i in range(len(c)):
        c[i].extend(shared)

    nf = int(sh + (12 - sh) / 5)
    tr_loader, val_loader, te_loader = forest_loader(batch_size=128, c=c, adv=adv, adv_valid=True)
    # tr_loader, val_loader, te_loader = adv_loader(batch_size=128, c=c, adv=adv, head='Forest10c3a_Sh' + str(sh),
    #                                               compress=True)
    for m in models:
        head2 = 'forest_Sh{0}_{1}'.format(sh, m)
        if m == 'FLRSH':
            model = FLRSH(feats=c, nc=5, classes=7)
        elif m == 'FLNSH':
            model = FLNSH(feats=c, nc=5, classes=7)
        elif m == 'FLRHZ':
            model = FLRHZ(feats=c, nf=[sh, nf-sh], nc=5, classes=7)
        else:
            raise Exception('Model not found.')
        opt = torch.optim.Adam(model.parameters())
        loss = nn.CrossEntropyLoss()

        cmab = fcmab(model, loss, opt, nc=5, n=100, c='mad', head=head2 + '5c2a_RandPert_Asynch1_MAD2',
                     adv_c=[0, 1], sync=False, ucb_c=2)
        cmab.train(tr_loader, val_loader, te_loader)
