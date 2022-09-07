from shap_model import *
from ni_data import *

use_gpu = False

batch_size = 128
shared = 'service'
swap = 0
cur_host = 0


def alpha(k):
    return 1e-4 / (1 + k)


train_loader, test_loader, model = ni_loader(batch_size=batch_size, cur_host=cur_host, shared=shared, swap=swap)


# model_id = "NI_SGD_mu0_K0_trained_model_best"
model_id = "fl11_model_service_swap0_0_best"
# s = shap(model, h + 'models/' + model_id + '.pt', alpha=alpha, use_gpu=use_gpu)
s = shap(model, './models/' + model_id + '.pt', alpha=alpha, use_gpu=use_gpu)
s.explainer(train_loader, test_loader)
# s.run(train_loader, test_loader)
