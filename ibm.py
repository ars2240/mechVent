from fcmab import *
from fnn import *
from floaders import *

sh = 341
if sh == 1:
    c0 = []
    c1 = []
elif sh == 85:
    c0 = []
    c1 = []
elif sh == 171:
    c0 = []
    c1 = []
elif sh == 255:
    c0 = []
    c1 = []
elif sh == 341:
    c0 = []
    c1 = []
else:
    raise Exception('Number of shared features not implemented.')
shared = [x for x in range(0, 122) if x not in c0 and x not in c1]
head = 'IBMU4_Sh' + str(sh)
adv = [*range(len(c0), len(c0)+len(shared))]
c0.extend(shared)
c1.extend(shared)

tr_loader, val_loader, te_loader = ibm_loader(batch_size=128, c0=c0, c1=c1, adv=adv, adv_valid=True, undersample=4)
raise Exception
# tr_loader, val_loader, te_loader = adv_forest_loader(batch_size=128, adv_valid=True, c0=c0, c1=c1, head=head + '_best')
# model = FLR(train_feat=[len(c0), len(c1)], nc=2, classes=7)
model = FLRSH(feats=[c0, c1], nc=2, classes=2)
opt = torch.optim.Adam(model.parameters())
loss = nn.CrossEntropyLoss()

cmab = fcmab(model, loss, opt, nc=2, n=100, c='mabLin', head=head + '_FLRSH_Adv', verbose=True)
cmab.train(tr_loader, val_loader, te_loader)
