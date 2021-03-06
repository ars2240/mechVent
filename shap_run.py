from shap_model import *
import os
import pickle
import sys
from lstm_model import LSTM
h = '/home/securedata/laurie/'
sys.path.insert(1, h)

import oversample_dict_dataset as ldd

with open(h + "/data/oversample_datasets/test.p", 'rb') as f:
    test = pickle.load(f)
with open(h + "/data/oversample_datasets/train.p", 'rb') as f:
    train = pickle.load(f)

os.chdir('/home/gluster/mechVent/')

use_gpu = True

pred_hrs = 4
batch_size = 128
stat = True
sequence_length = 48
hidden_size = 64
num_layers = 1

model_prefix = "LSTM_rsample_os3_s133_normstat_"
num_epochs = 150
lr = 0.001
model_id = model_prefix + str(num_epochs) + '_' + str(hidden_size) + \
    '_' + str(lr).split('.')[1]


train_data = ldd.build_dataset(train, pred_hrs, batch_size, stat)
test_data = ldd.build_dataset(test, pred_hrs, batch_size, stat)
model = LSTM(sequence_length, hidden_size, num_layers)


def alpha(k):
    return 1e-4 / (1 + k)


s = shap(model, h + '/results/oversample_results/' + model_id + '/' + model_id + '.pt', alpha=alpha, use_gpu=use_gpu)
s.explainer(train_data, test_data)
# s.run(train_loader, test_loader)
