from shap_model import *
import os
import sys
sys.path.insert(1, '/Users/adamsandler/Documents/Northwestern/Research/Optimization with Bounded Eigenvalues/')

from usps_data import get_train_valid_loader, get_test_loader, CNN

os.chdir('/Users/adamsandler/Documents/Northwestern/Research/Optimization with Bounded Eigenvalues/')

batch_size = 128

train_loader, valid_loader = get_train_valid_loader(batch_size=batch_size, augment=False)
test_loader = get_test_loader(batch_size=batch_size)
model = CNN()


def alpha(k):
    return 1e-4 / (1 + k)


s = shap(model, './models/USPS_SGD_mu0_K0_trained_model_best.pt', alpha=alpha)
s.explainer(train_loader, test_loader)
# s.run(train_loader, test_loader)
