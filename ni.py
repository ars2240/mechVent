from fcmab import *
from fnn import *
from floaders import *


c0 = [37]
c1 = [21]
# shared = [x for x in range(0, 118) if x not in c0 and x not in c1]
shared = [0, 5, 19, 26, 40]
adv = [*range(len(c0), len(c0)+len(shared))]
c0.extend(shared)
c1.extend(shared)

# tr_loader, val_loader, te_loader = ni_loader(batch_size=128, c0=c0, c1=c1, adv=adv, adv_valid=True)
tr_loader, val_loader, te_loader = adv_forest_loader(batch_size=128, adv_valid=True, c0=c0, c1=c1, head='NIShare5_3G4B_best')
# model = FLR(train_feat=[len(c0), len(c1)], nc=2, classes=7)
model = FLRSH(feats=[c0, c1], nc=2, classes=2)
opt = torch.optim.Adam(model.parameters())
loss = nn.CrossEntropyLoss()

cmab = fcmab(model, loss, opt, nc=2, n=100, c='mabLin', head='niRed_FLRSH_Adv_5sh_3G4B', verbose=True)
cmab.train(tr_loader, val_loader, te_loader)
