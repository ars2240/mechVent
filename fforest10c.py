from fcmab import *
from fnn import *
from floaders import *

for sh in range(2, 13, 10):
    if sh == 2:
        c = [[6], [9], [4], [8], [1], [5], [*range(14, 54)], [2], [*range(10, 14)], [3]]
        shared = [0, 7]
    elif sh == 12:
        c = [[], [], [], [], [], [], [], [], [], []]
        shared = [*range(0, 54)]
    else:
        raise Exception('Number of shared features not implemented.')
    adv = [*range(len(c[0]), len(c[0])+len(shared))]
    for i in range(len(c)):
        c[i].extend(shared)
    print(c)

    tr_loader, val_loader, te_loader = forest_loader(batch_size=128, c=c, adv=adv, adv_valid=True)
    # tr_loader, val_loader, te_loader = adv_forest_loader(batch_size=128, adv_valid=True, c0=c0, c1=c1, head='advLogReg2AdamRandInitShare12_best')
    model = FLNSH(feats=c, nc=10, classes=7)
    opt = torch.optim.Adam(model.parameters())
    loss = nn.CrossEntropyLoss()

    cmab = fcmab(model, loss, opt, nc=10, n=100, c='mabLin', head='forest_FLNSH10c_Pert_Sh' + str(sh), verbose=True)
    cmab.train(tr_loader, val_loader, te_loader)
