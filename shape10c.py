from fcmab import *
from fnn import *
from floaders import *
from itertools import chain

for sh in range(4, 25, 10):
    if sh == 4:
        c = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19]]
    elif sh == 14:
        c = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
    elif sh == 24:
        c = [[], [], [], [], [], [], [], [], [], []]
    else:
        raise Exception('Number of shared features not implemented.')
    shared = [x for x in range(0, 24) if x not in chain(*c)]
    adv = {0: [*range(len(c[0]), len(c[0]) + len(shared))], 1: [*range(len(c[1]), len(c[1]) + len(shared))],
           2: [*range(len(c[2]), len(c[2]) + len(shared))]}
    for i in range(len(c)):
        c[i].extend(shared)

    nf = int(sh + (24 - sh) / 10)
    imgsz = 3 * (137 ** 2)
    tr_loader, val_loader, te_loader = shape_loader(batch_size=128, c=c, adv=adv, adv_valid=True)

    for m in ['FLRHZ']:
    # for m in ['FLRSH', 'FLNSH']:
        head2 = 'shape_Sh{0}_{1}'.format(sh, m)
        if m == 'FLRSH':
            model = FLRSH(feats=c, nc=10, classes=13)
        elif m == 'FLNSH':
            model = FLNSH(feats=c, nc=10, classes=13)
        elif m == 'FLRHZ':
            model = FLRHZ(feats=c, nf=[sh, nf-sh], m=imgsz, nc=10, classes=13)
        else:
            raise Exception('Model not found.')
        opt = torch.optim.Adam(model.parameters())
        loss = nn.CrossEntropyLoss()

        cmab = fcmab(model, loss, opt, nc=10, n=100, c='mad', head=head2 + '10c3a_RandPert_Asynch1_MAD2',
                     adv_c=[0, 1, 2], sync=False, ucb_c=2, embed_mu=1)
        cmab.train(tr_loader, val_loader, te_loader)
