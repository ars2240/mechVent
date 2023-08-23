from fcmab import *
from fnn import *
from floaders import *


# c0 = [0, 9, 10, 25, 36]
c0 = [1, 14, *range(41, 107)]
# c1 = [6, 16, 31, 35, 38]
c1 = [8, 30, 33, *range(107, 118)]
# shared = [x for x in range(0, 118) if x not in c0 and x not in c1]
shared = []
adv = [*range(len(c0), len(c0)+len(shared))]
c0.extend(shared)
c1.extend(shared)

tr_loader, val_loader, te_loader = ni_loader(batch_size=128, c0=c0, c1=c1, adv=adv, adv_valid=True)
# model = FLR(train_feat=[len(c0), len(c1)], nc=2, classes=7)
model = FLRSH(feats=[c0, c1], nc=2, classes=2)
opt = torch.optim.Adam(model.parameters())
loss = nn.CrossEntropyLoss()

cmab = fcmab(model, loss, opt, nc=2, n=100, c='mabLin', head='niRed_FLRSH_RandPert_0sh', verbose=True)
cmab.train(tr_loader, val_loader, te_loader)
