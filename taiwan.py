from fcmab import *
from fnn import *
from floaders import *


c0 = [50, 52, 66, 68, 93]
c1 = [46, 57, 61, 87, 91]
# shared = [x for x in range(0, 118) if x not in c0 and x not in c1]
shared = [x for x in range(0, 95) if x not in c0 and x not in c1]
adv = [*range(len(c0), len(c0)+len(shared))]
c0.extend(shared)
c1.extend(shared)

tr_loader, val_loader, te_loader = taiwan_loader(batch_size=128, c0=c0, c1=c1, adv=adv, adv_valid=True)
model = FLR(train_feat=[len(c0), len(c1)], nc=2, classes=2)
opt = torch.optim.Adam(model.parameters())
loss = nn.CrossEntropyLoss()

cmab = fcmab(model, loss, opt, nc=2, n=100, c='mabLin', head='taiwan_FLR_RandPert_85sh', verbose=True)
cmab.train(tr_loader, val_loader, te_loader)
