from fcmab import *
from fnn import *
from floaders import *

crop = 0
pad = False

tr_loader, val_loader, te_loader = cifar_loader(batch_size=128, crop=crop, pad=pad)
size = 32*32*3 if pad else 32*(32-crop)*3
# model = FLR(train_feat=[size, size], nc=2, classes=10)
model = FLCNN(classes=10)
opt = torch.optim.Adam(model.parameters())
loss = nn.CrossEntropyLoss()

cmab = fcmab(model, loss, opt, nc=2, n=100, c='mabLin', head='cifar_FLCNN_GausBlur3_0crop', verbose=True)
cmab.train(tr_loader, val_loader, te_loader)
