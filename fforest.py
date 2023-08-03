from fcmab import *
from fnn import *
from floaders import *


c0 = [1, 6, 8]
c1 = [4, 5, 9]
shared = [x for x in range(0, 54) if x not in c0 and x not in c1]
adv = [*range(len(c0), len(c0)+len(shared))]
c0.extend(shared)
c1.extend(shared)

tr_loader, val_loader, te_loader = forest_loader(batch_size=128, c0=c0, c1=c1, adv=adv, adv_valid=True)
# tr_loader, val_loader, te_loader = adv_forest_loader(batch_size=128, split=51, head='advLogReg2AdamIgnRandInit_best')
# tr_loader, val_loader, te_loader = adv_forest_loader(batch_size=128, split=51, adv_valid=True, c0=c0, c1=c1, head='advLogReg2AdamIgnRandInit_best')
# model = FLN(train_feat=[51, 51], nc=2, h1=4, classes=7)
# model = FLR(train_feat=[51, 51], nc=2, classes=7)
model = FLRSH(feat=[c0, c1], nc=2, classes=7)
opt = torch.optim.Adam(model.parameters())
loss = nn.CrossEntropyLoss()

# cmab = fcmab(model, loss, opt, nc=2, n=100, c='mabLin', head='forest_mean_LR_actadv_nmap2_mabLin', adversarial=0, m=1, ab=1)
# cmab = fcmab(model, loss, opt, nc=2, n=100, c='mabLin', head='forest_mean_LR_actAdvAdam10_step_mabLin', adv_epoch=10, adv_opt='adam', adv_step=.001, adversarial=0, verbose=True)
# cmab = fcmab(model, loss, opt, nc=2, n=100, c='mabLin', head='forest_mean_LR_nmap2_mabLin_c.6', m=1, ab=1, ucb_c=.6)
cmab = fcmab(model, loss, opt, nc=2, n=100, c='mabLin', head='forest_FLRSH_RandPert', verbose=True)
cmab.train(tr_loader, val_loader, te_loader)
# cmab.load(head=cmab.head + '_best', info=True)
