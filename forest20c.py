from fcmab import *
from fnn import *
from floaders import *
from itertools import chain

models = ['FLRSH']

for sh in [12]:
    if sh == 12:
        c = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    else:
        raise Exception('Number of shared features not implemented.')
    shared = [x for x in range(0, 54) if x not in chain(*c)]
    adv = {0: [*range(len(c[0]), len(c[0]) + len(shared))], 1: [*range(len(c[1]), len(c[1]) + len(shared))],
           2: [*range(len(c[2]), len(c[2]) + len(shared))], 3: [*range(len(c[3]), len(c[3]) + len(shared))],
           4: [*range(len(c[4]), len(c[4]) + len(shared))], 5: [*range(len(c[5]), len(c[5]) + len(shared))]}
    for i in range(len(c)):
        c[i].extend(shared)

    nf = int(sh + (12 - sh) / 20)
    tr_loader, val_loader, te_loader = forest_loader(batch_size=128, c=c, adv=adv, adv_valid=True)
    # tr_loader, val_loader, te_loader = adv_loader(batch_size=128, c=c, adv=adv, head='Forest10c3a_Sh' + str(sh),
    #                                               compress=True)
    for m in models:
        head2 = 'forest_Sh{0}_{1}'.format(sh, m)
        if m == 'FLRSH':
            model = FLRSH(feats=c, nc=20, classes=7)
        elif m == 'FLNSH':
            model = FLNSH(feats=c, nc=20, classes=7)
        elif m == 'FLRHZ':
            model = FLRHZ(feats=c, nf=[sh, nf-sh], nc=20, classes=7)
        else:
            raise Exception('Model not found.')
        opt = torch.optim.Adam(model.parameters())
        loss = nn.CrossEntropyLoss()

        cmab = fcmab(model, loss, opt, nc=20, n=100, c='allgood', head=head2 + '20c6a_allgood_RandPert_Reset',
                     adv_c=[*range(6)], fix_reset=True)
        cmab.train(tr_loader, val_loader, te_loader)
