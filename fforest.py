from fcmab import *
from fnn import *
from floaders import *


def s(e):
    return 1/(e+1)


# train_loader, valid_loader, test_loader = forest_loader(batch_size=128)
train_loader, valid_loader, test_loader = adv_forest_loader(batch_size=128, split=51, head='advLogReg2AdamIgn')
# model = FLN(train_feat=[14, 44], nc=2, h1=10, classes=7)
model = FLR(train_feat=[51, 51], nc=2, classes=7)
opt = torch.optim.Adam(model.parameters())
loss = nn.CrossEntropyLoss()

# cmab = fcmab(model, loss, opt, nc=2, n=100, c='mabLin', head='forest_mean_LR_actadv_nmap2_mabLin', adversarial=0, m=1, ab=1)
# cmab = fcmab(model, loss, opt, nc=2, n=100, c='mabLin', head='forest_mean_LR_actAdvAdam10_step_mabLin', adv_epoch=10, adv_opt='adam', adv_step=.001, adversarial=0, verbose=True)
# cmab = fcmab(model, loss, opt, nc=2, n=100, c='mabLin', head='forest_mean_LR_nmap2_mabLin_c.6', m=1, ab=1, ucb_c=.6)
cmab = fcmab(model, loss, opt, nc=2, n=100, c='mabLin', head='forest_mean_LR_mabLin_adv2AdamIgn', verbose=True)
cmab.train(train_loader, valid_loader, test_loader)
