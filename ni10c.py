from fcmab import *
from fnn import *
from floaders import *
from itertools import chain

sh = 1
if sh == 1:
    c = [[16, 19, 25, 26], [5, 8, 12, 24], [3, 9, 29, *range(41, 111)], [7, 17, 32, 35], [14, 36, 37, *range(38, 41)],
         [4, 15, 20, 23], [2, 13, 18, 28], [1, 6, 30, 31], [0, 27, 33, 34], [10, 11, 21, *range(111, 122)]]
elif sh == 11:
    c = [[16, 25, 26], [5, 8, 12], [3, 9, *range(41, 111)], [7, 17, 35], [14, 37, *range(38, 41)], [15, 20, 23],
         [2, 13, 28], [1, 6, 31], [0, 27, 34], [10, 11, 21]]
elif sh == 21:
    c = [[16, 25], [5, 12], [3, 9], [7, 17], [14, 37], [15, 23], [2, 13], [1, 31], [0, 34], [10, 11]]
elif sh == 31:
    c = [[16], [5], [3], [17], [14], [15], [13], [1], [0], [10]]
elif sh == 41:
    c = [[], [], [], [], [], [], [], [], [], []]
else:
    raise Exception('Number of shared features not implemented.')
shared = [x for x in range(0, 122) if x not in chain(*c)]
head = 'NI+Share' + str(sh)
adv = [*range(len(c[0]), len(c[0])+len(shared))]
for i in range(len(c)):
    c[i].extend(shared)
print(c)

tr_loader, val_loader, te_loader = ni_loader(batch_size=128, c=c, adv=adv, adv_valid=True)
# tr_loader, val_loader, te_loader = adv_forest_loader(batch_size=128, adv_valid=True, c0=c0, c1=c1, head=head + '_best')
# model = FLR(train_feat=[len(c0), len(c1)], nc=2, classes=7)
model = FLRSH2(feats=c, nc=10, classes=2)
opt = torch.optim.Adam(model.parameters())
loss = nn.CrossEntropyLoss()

cmab = fcmab(model, loss, opt, nc=10, n=100, c='mabLin', head=head + '_FLRSH_RandPert', verbose=True)
cmab.train(tr_loader, val_loader, te_loader)
