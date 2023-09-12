from fcmab import *
from fnn import *
from floaders import *


c0 = [5, 6]
c1 = [1, 9]
c2 = [4, 8]
shared = [x for x in range(0, 54) if x not in c0 and x not in c1 and x not in c2]
adv = [*range(len(c0), len(c0)+len(shared))]
c0.extend(shared)
c1.extend(shared)
c2.extend(shared)

tr_loader, val_loader, te_loader = forest_loader(batch_size=128, c0=c0, c1=c1, c2=c2, adv=adv, nc=3, adv_valid=True)
# tr_loader, val_loader, te_loader = adv_forest_loader(batch_size=128, adv_valid=True, c0=c0, c1=c1, head='advLogReg2AdamRandInitShare12_best')
model = FLRSH(feats=[c0, c1, c2], nc=3, classes=7)
opt = torch.optim.Adam(model.parameters())
loss = nn.CrossEntropyLoss()

cmab = fcmab(model, loss, opt, nc=3, n=100, c='mabLin', head='forest_FLRSH3c_Pert_Sh6', verbose=True)
cmab.train(tr_loader, val_loader, te_loader)
# cmab.load(head=cmab.head + '_best', info=True)
