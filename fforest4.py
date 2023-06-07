from fcmab import *
from fnn import *
from floaders import *


c0 = [1, 6, 8]
c1 = [4, 5, 9]
shared = [x for x in range(0, 54) if x not in c0 and x not in c1]
adv = [*range(len(c0), len(c0)+len(shared))]
c0.extend(shared)
c1.extend(shared)

head = 'advLogReg2AdamIgnRandInit_best'

model = FLR(train_feat=[len(c0), len(c1)], nc=2, classes=7)
opt = torch.optim.Adam(model.parameters())
loss = nn.CrossEntropyLoss()

tr_load, val_load, te_load = forest_loader(batch_size=128, c0=c0, c1=c1, adv=adv, adv_valid=True)
cmab = fcmab(model, loss, opt, nc=2, n=100, c='mabLin', head='forest_mean_LR_mabLin2RandPert', verbose=True)
cmab.train(tr_load, val_load, te_load)

tr_load, val_load, te_load = adv_forest_loader(batch_size=128, split=len(c0), adv_valid=True, c0=c0, c1=c1, head=head)
cmab = fcmab(model, loss, opt, nc=2, n=100, c='mabLin', head='forest_mean_LR_mabLin2Adv', verbose=True)
cmab.train(tr_load, val_load, te_load)

tr_load, val_load, te_load = forest_loader(batch_size=128, c0=c0, c1=c1, adv=adv, adv_valid=False)
cmab = fcmab(model, loss, opt, nc=2, n=100, c='mabLin', head='forest_mean_LR_mabLin2RandPert_ValGood', verbose=True)
cmab.train(tr_load, val_load, te_load)

tr_load, val_load, te_load = adv_forest_loader(batch_size=128, split=len(c0), adv_valid=False, c0=c0, c1=c1, head=head)
cmab = fcmab(model, loss, opt, nc=2, n=100, c='mabLin', head='forest_mean_LR_mabLin2Adv_ValGood', verbose=True)
cmab.train(tr_load, val_load, te_load)
