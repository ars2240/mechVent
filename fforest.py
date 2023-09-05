from fcmab import *
from fnn import *
from floaders import *


c0 = [0, 1, 2, 6, 8, *range(10, 14)]
c1 = [3, 4, 5, 7, 9, *range(14, 54)]
shared = [x for x in range(0, 54) if x not in c0 and x not in c1]
adv = [*range(len(c0), len(c0)+len(shared))]
c0.extend(shared)
c1.extend(shared)

tr_loader, val_loader, te_loader = forest_loader(batch_size=128, c0=c0, c1=c1, adv=adv, adv_valid=True)
# tr_loader, val_loader, te_loader = adv_forest_loader(batch_size=128, split=51, head='advLogReg2AdamIgnRandInit_best')
# tr_loader, val_loader, te_loader = adv_forest_loader(batch_size=128, adv_valid=True, c0=c0, c1=c1, head='advLogReg2AdamRandInitShare12_best')
# model = FLN(train_feat=[51, 51], nc=2, h1=4, classes=7)
# model = FLR(train_feat=[51, 51], nc=2, classes=7)
raise Exception
model = FLNSH(feats=[c0, c1], nc=2, classes=7)
opt = torch.optim.Adam(model.parameters())
loss = nn.CrossEntropyLoss()

# cmab = fcmab(model, loss, opt, nc=2, n=100, c='mabLin', head='forest_mean_LR_actadv_nmap2_mabLin', adversarial=0, m=1, ab=1)
# cmab = fcmab(model, loss, opt, nc=2, n=100, c='mabLin', head='forest_mean_LR_actAdvAdam10_step_mabLin', adv_epoch=10, adv_opt='adam', adv_step=.001, adversarial=0, verbose=True)
# cmab = fcmab(model, loss, opt, nc=2, n=100, c='mabLin', head='forest_mean_LR_nmap2_mabLin_c.6', m=1, ab=1, ucb_c=.6)
cmab = fcmab(model, loss, opt, nc=2, n=100, c='mabLin', head='forest_FLNSH_Pert_Sh0', verbose=True)
cmab.train(tr_loader, val_loader, te_loader)
# cmab.load(head=cmab.head + '_best', info=True)
