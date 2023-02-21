from fcmab import *
from fnn import *
from floaders import *


train_loader, valid_loader, test_loader = forest_loader(batch_size=128)
model = FLN(train_feat=[14, 44], nc=2)
opt = torch.optim.Adam(model.parameters())
loss = nn.CrossEntropyLoss()
cmab = fcmab(model, loss, opt, nc=2, head='forest')
cmab.train(train_loader, valid_loader, test_loader)
