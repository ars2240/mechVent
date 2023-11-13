from fcmab import *
from fnn import *
from floaders import *
from itertools import chain

for sh in range(2, 13, 10):
    if sh == 2:
        c = [[6], [9], [4], [8], [1], [5], [*range(14, 54)], [2], [*range(10, 14)], [3]]
    elif sh == 12:
        c = [[], [], [], [], [], [], [], [], [], []]
    else:
        raise Exception('Number of shared features not implemented.')
    shared = [x for x in range(0, 54) if x not in chain(*c)]
    adv = {0: [*range(len(c[0]), len(c[0]) + len(shared))], 1: [*range(len(c[1]), len(c[1]) + len(shared))],
           2: [*range(len(c[2]), len(c[2]) + len(shared))]}
    for i in range(len(c)):
        c[i].extend(shared)
    print(c)

    # tr_loader, val_loader, te_loader = forest_loader(batch_size=128, c=c, adv=adv, adv_valid=True)
    tr_loader, val_loader, te_loader = adv_loader(batch_size=128, c=c, adv=adv, head='Forest10c3a_Sh' + str(sh),
                                                  compress=True)
    model = FLNSH(feats=c, nc=10, classes=7)
    opt = torch.optim.Adam(model.parameters())
    loss = nn.CrossEntropyLoss()

    cmab = fcmab(model, loss, opt, nc=10, n=100, c='mabLin', head='forest_FLNSH10c3a_AdvHztl_Sh' + str(sh),
                 adv_c=[0, 1, 2], verbose=True)
    cmab.train(tr_loader, val_loader, te_loader)
