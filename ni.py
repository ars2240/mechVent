from fcmab import *
from fnn import *
from floaders import *

for sh in [1, 11, 21, 31, 41]:
    if sh == 1:
        c0 = [0, 1, 6, 7, 14, 16, 17, 19, 25, 26, 27, 30, 31, 32, 33, 34, 35, 36, 37, *range(38, 41)]
        c1 = [2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 15, 18, 20, 21, 23, 24, 28, 29, *range(41, 122)]
    elif sh == 11:
        c0 = [0, 1, 6, 7, 14, 16, 17, 25, 26, 27, 31, 34, 35, 37, *range(38, 41)]
        c1 = [2, 3, 5, 8, 9, 10, 11, 12, 13, 15, 20, 21, 23, 28, *range(41, 111)]
    elif sh == 21:
        c0 = [0, 1, 7, 14, 16, 17, 25, 31, 34, 37]
        c1 = [2, 3, 5, 9, 10, 11, 12, 13, 15, 23]
    elif sh == 31:
        c0 = [0, 1, 14, 16, 17]
        c1 = [3, 5, 10, 13, 15]
    elif sh == 41:
        c0 = []
        c1 = []
    else:
        raise Exception('Number of shared features not implemented.')
    shared = [x for x in range(0, 122) if x not in c0 and x not in c1]
    head = 'NI+Share' + str(sh)
    adv = [*range(len(c0), len(c0)+len(shared))]
    c0.extend(shared)
    c1.extend(shared)

    tr_loader, val_loader, te_loader = ni_loader(batch_size=128, c=[c0, c1], adv=adv, adv_valid=True)
    # tr_loader, val_loader, te_loader = adv_forest_loader(batch_size=128, adv_valid=True, c0=c0, c1=c1, head=head + '_best')
    # model = FLR(train_feat=[len(c0), len(c1)], nc=2, classes=7)
    model = FLRSH(feats=[c0, c1], nc=2, classes=2)
    opt = torch.optim.Adam(model.parameters())
    loss = nn.CrossEntropyLoss()

    cmab = fcmab(model, loss, opt, nc=2, n=100, c='mabLin', head=head + '_FLRSH_RandPert', verbose=True)
    cmab.train(tr_loader, val_loader, te_loader)
