from fcmab import *
from fnn import *
from floaders import *


train_loader, valid_loader, test_loader = cifar_loader(batch_size=128)
model = FLCNN()
opt = torch.optim.Adam(model.parameters())
loss = nn.CrossEntropyLoss()
cmab = fcmab(model, loss, opt, nc=2, head='cifar10')
cmab.train(train_loader, valid_loader, test_loader)
