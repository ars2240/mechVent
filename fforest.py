from fcmab import *
from fnn import *
from floaders import *

train_loader, valid_loader, test_loader = forest_loader(batch_size=128)
# train_loader, valid_loader, test_loader = adv_forest_loader(batch_size=128, split=47, head='advLogReg2')
# model = FLN(train_feat=[14, 44], nc=2, h1=10, classes=7)
model = FLR(train_feat=[47, 47], nc=2, classes=7)
opt = torch.optim.Adam(model.parameters())
loss = nn.CrossEntropyLoss()
# cmab = fcmab(model, loss, opt, nc=2, n=100, c='mabLin', head='forest_mean_LR_actadv_nmap2_mabLin', adversarial=0, m=1, ab=1)
cmab = fcmab(model, loss, opt, nc=2, n=100, c='mabLinHyb', head='forest_mean_LR_actadv_mabLinHyb', adversarial=0, verbose=True)
# cmab = fcmab(model, loss, opt, nc=2, n=100, c='mabLin', head='forest_mean_LR_nmap2_mabLin', m=1, ab=1)
# cmab = fcmab(model, loss, opt, nc=2, n=100, c='mabLinHyb', head='forest_mean_LR_mabLinHyb', verbose=True)
cmab.train(train_loader, valid_loader, test_loader)
