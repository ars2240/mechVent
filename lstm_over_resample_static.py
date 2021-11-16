import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
import pickle
# from experiment_tracking import ExperimentTracker
import csv
# file with useful modeling functions
import lamk_data_prep as ldp
import oversample_dict_dataset as ldd
import lamk_nnfuncs as nnfuncs
from sklearn.metrics import auc, roc_curve
import csv_tracker as track
from datetime import datetime


# user inputs - script is designed so that you can just make changes in this seection and then re-run for testing purposes
# data directory where the time series data will be loaded from
time_data_dir = "/home/securedata/data/imputed/"
# data directory where the static data will be loaded from
static_data_dir = "/home/securedata/data/raw_data/"
model_dir = '/home/securedata/data/'
# model prefix -- will appear in tracking file and artefact file names
model_prefix = "LSTM_rsample_os3_s133_normstat_"
# whether to re-make the negative and positive sample data or load existing from file
# if changing train, pred, or before_intub hours should NOT use "existing" 
# set to "new" to save out the new positive and negative samples generated in this run
neg_pos_save = "new"
# same deal as negative/positive data - can set to new to save out or existing to load existing save of train/val/test pickled dictionary data
train_val_test_save = "new"
# oversample amount - must be an int, the positive cases in the training data will be replicated this number of times
oversample_by = 3
# flag whether to use resampling during training
resample_flag = True
# flag whether to use static predictors or not - this setting must match whether you're using the static or non static version of the script (the model architecture is also changed if using statis and isn't branched yet with if logic -- the  two options are separate lstm scripts)
# this script (oversampling) DOES expect static predictors
stat = True
# number of hours for the prediction period (yes, it's confusingly named)
train_hrs = 4
# number of hours for the data collection period
pred_hrs = 4
# gap between end of data collection window and first possible intubation time
hrs_before_intub = .5
# random seed to use in places where a seed is possible
seed = 133
# proportion of data to use for training, validation, and test (must sum to 1)
train_prop = .6
val_prop = .2
test_prop = .2
# batch size to use in nn training
batch_size = 128
# list of column names for time series and static  predictors
time_preds = ['rrvi_norm', 'rr_mean_norm', 'hrvi_norm', 'hr_mean_norm', 'sat_mean']
static_preds = ['malignancy', 'transpl', 'sum_comorb_ccc']
torch.manual_seed(seed)
torch.manual_seed(seed)
# model architecture
# num obs in training sequence - should equal pred_hrs * 12 (12 = 60/5 min)
sequence_length = 48
# number of hidden layers in network - to change this may need to make other changes in LSTM class or in model training
num_layers = 1
# number of nodes in hidden layers in network
hidden_size = 128
# learning rate
lr = 0.001
# training epochs
num_epochs = 150

print("loading data at " + str(datetime.now()))
if neg_pos_save != "existing":
    # load and reshape data
    # make the full predictor df
    pred_df = ldd.make_pred(time_data_dir, static_data_dir)
    # filter based on the hour limits indicated 
    pos_samples_f, neg_samples_f = ldd.make_filtered_df(pred_df, pred_hrs, train_hrs, hrs_before_intub)
    if neg_pos_save == "new":
        pos_samples_f.to_csv("pos_samples_f.csv", index = False)
        neg_samples_f.to_csv("neg_samples_f.csv", index = False)
else:
    pos_samples_f = pd.read_csv("pos_samples_f.csv")
    neg_samples_f = pd.read_csv("neg_samples_f.csv")

# train test split
print("train val test split at " + str(datetime.now()))
if train_val_test_save != "existing":
    train, val, test = ldd.make_train_val_test(pos_samples_f, neg_samples_f, seed, train_prop, val_prop, test_prop, time_preds, static_preds, pred_hrs, resample_flag, oversample_by)
    if train_val_test_save == "new":
        with open("test.p",'wb') as f:
            pickle.dump(test, f)
        with open("train.p",'wb') as f:
            pickle.dump(train,f)
        with open("val.p",'wb') as f:
            pickle.dump(val,f)
else:
    with open("test.p",'rb') as f:
        test = pickle.load(f)
    with open("train.p",'rb') as f:
        train = pickle.load(f)
    with open("val.p",'rb') as f:
        val = pickle.load(f)

# creating dataset
print("build train pytorch dataset at " + str(datetime.now() ))
train_data = ldd.build_dataset(train, pred_hrs, batch_size, stat)
print("build val pytorch dataset at " + str(datetime.now()))
val_data = ldd.build_dataset(val, pred_hrs, batch_size, stat)
# combine val and train: https://www.geeksforgeeks.org/python-merging-two-dictionaries/
full_train_data = ldd.build_dataset({**train, **val}, pred_hrs, batch_size, stat)
print("build test pytorch dataset at " + str(datetime.now()))
test_data = ldd.build_dataset(test, pred_hrs, batch_size, stat)

# data and model go to GPU
device = torch.device('cuda')

# epochs, hiddensize, LR
model_id = model_prefix + str(num_epochs) + '_' + str(hidden_size) + \
    '_' + str(lr).split('.')[1]

tracker = track.CSVTracker('home/securedata/results/experiment_tracking.csv')

tracker.is_model_unique(model_id)

# creating a model_id folder in which plots can be saved
if not os.path.isdir('results/' + model_id):
    os.mkdir('results/' + model_id)


class LSTM(nn.Module):

    def __init__(self, sequence_length, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        # 5 predictors
        self.input_size = 5
        # binary classification
        self.num_classes = 2
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size*sequence_length, self.num_classes)
        # references: https://discuss.pytorch.org/t/sequential-and-static-features-combined-in-lstm-architecture/91115
        self.fc2 = nn.Linear(3,self.num_classes)
        self.fc3 = nn.Linear(4,self.num_classes)

    def init_hidden(self, x):
        self.batch_size = x.size()[0]
        self.hidden_cell = (torch.zeros(num_layers, self.batch_size,
                                        hidden_size, device=device),
                            torch.zeros(num_layers, self.batch_size,
                                        hidden_size, device=device))

    def forward(self, x_time, x_stat):
        # input data x
        # can use multiple inputs to forward method: https://discuss.pytorch.org/t/multiple-input-model-architecture/19754
        # for the view call: batch size, sequence length, cols
        lstm_out, self.hidden_cell = self.lstm(x_time.view(self.batch_size,
                                                      x_time.size()[1], -1),
                                               self.hidden_cell)
       # stat_out = stat_input.reshape(self.batch_size,-1)      

        stat_preds = self.fc2(x_stat.reshape(self.batch_size,-1))
        time_preds = self.fc1(lstm_out.reshape(self.batch_size,-1))
        preds=torch.cat((stat_preds,time_preds),1)
        preds = self.fc3(preds)
        return preds.view(self.batch_size, -1)

# initializing model
model = LSTM(sequence_length=sequence_length, hidden_size=hidden_size,
             num_layers=num_layers)

# send to gpu
model.cuda()

# setting optimizer and loss
# weights inversely proportional to class frequency
# class_weights = nnfuncs.get_class_weights(y_train.view(-1))
# class_weights = class_weights.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss = nn.CrossEntropyLoss()

# initializing roc value and list to track train and val losses after each epoch
prev_roc = 0
# saving optimal epoch number
opt_epoch = 0
train_loss = []
val_loss = []
print("training model at " + str(datetime.now()))
for epoch in range(num_epochs):
    for batch_n, (X_time, X_stat, y) in enumerate(train_data):
        X_time = X_time.float().to(device)
        X_stat = X_stat.float().to(device)
        # print(X.shape)
        # cross entropy loss takes an int as the target which corresponds
        # to the index of the target class
        # output should be of shape (batch_size, num_classes)
        y = y.long().to(device)

        # zero out the optimizer gradient
        # no dependency between samples
        optimizer.zero_grad()
        model.init_hidden(X_time)
        y_pred = model(X_time, X_stat)
        # print(y_pred.size())
        # print(y.size())
        batch_loss = loss(y_pred, y.view(-1))
        batch_loss.backward()
        optimizer.step()

    # training and validation outputs, predictions of class 1, labels
    train_output, train_pred, train_label = nnfuncs.get_preds_labels(model,
                                                                     train_data,
                                                                     device, True)
    val_output, val_pred, val_label = nnfuncs.get_preds_labels(model,
                                                               val_data,
                                                               device, True)

    # calculating accuracy at 0.5 threshold
    acc = nnfuncs.model_accuracy(val_pred, val_label, 0.5)
    # label must be a long in crossentropy loss calc
    epoch_loss_train = loss(train_output, train_label.long().view(-1))
    epoch_loss_val = loss(val_output, val_label.long().view(-1))

    # must graph the epoch losses to check for convergence
    # appending loss vals to list after each epoch
    train_loss.append(epoch_loss_train.item())
    val_loss.append(epoch_loss_val.item())

    # must put tensors on the cpu to convert to numpy array
    # getting validation set roc and saving model
    # if roc improves
    fpr, tpr, _ = roc_curve(val_label.cpu(), val_pred.cpu())
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    if roc_auc > prev_roc:
        # save over roc
        prev_roc = roc_auc
        opt_epoch = epoch
        roc_path = 'results/' + model_id + '/' + model_id + '_val_roc.png'
        nnfuncs.plot_roc(fpr, tpr, roc_auc, roc_path)
    print(str(datetime.now()))
    print('Epoch {0} Validation Set Accuracy: {1}'.format(epoch, acc))
    print('Validation Set AUROC {}'.format(roc_auc))


nnfuncs.plot_loss(model_id, train_loss, val_loss)

# test results
train_best_model = LSTM(sequence_length=sequence_length, hidden_size=hidden_size,
                        num_layers=num_layers)
# need to reinitialize the optimizer to train again
best_model_optim = torch.optim.Adam(train_best_model.parameters(), lr=lr)
train_best_model.to(device)

# training the best model
if stat:
    nnfuncs.stat_train_best_model(train_best_model, model_id, full_train_data, opt_epoch, best_model_optim, loss, device=device)
else:
    nnfuncs.train_best_model(train_best_model, model_id, full_train_data, opt_epoch, best_model_optim, loss, device=device)

# loading the best model
best_model = LSTM(sequence_length=sequence_length, hidden_size=hidden_size,
                  num_layers=num_layers) # must first initialize an empty model
best_model_path = 'results/' + model_id + '/' + model_id + '.pt'
best_model_sd = torch.load(best_model_path)
best_model.load_state_dict(best_model_sd)
best_model.to(device)

test_output, test_pred, test_label = nnfuncs.get_preds_labels(best_model,
                                                              test_data,
                                                              device, True)


test_acc = nnfuncs.model_accuracy(test_pred, test_label, 0.5)
test_precision, test_recall, test_f1 = nnfuncs.precision_recall_f1(test_label,
                                                                   test_pred,
                                                                   0.5)


pr_path = 'results/' + model_id + '/' + model_id + '_precision_recall.png'
nnfuncs.plot_test_precision_recall(pr_path, test_label, test_pred)


# plotting and saving ROC curve as well
test_fpr, test_tpr, _ = roc_curve(test_label.cpu(), test_pred.cpu())
test_roc_auc = auc(test_fpr, test_tpr)

test_roc_path = 'results/' + model_id + '/' + model_id + '_test_roc.png'
nnfuncs.plot_roc(test_fpr, test_tpr, test_roc_auc, test_roc_path)

model.to('cpu')
torch.save(model.state_dict(), 'results/' + model_id + '/' + model_id + '_model.pt')

tracking_dict = {}
tracking_dict['model_type'] = model_id.split('_')[0]
tracking_dict['model_id'] = model_id
tracking_dict['epochs'] = num_epochs
tracking_dict['learning_rate'] = lr
tracking_dict['hidden_size'] = hidden_size
tracking_dict['accuracy'] = test_acc
tracking_dict['precision'] = test_precision
tracking_dict['recall'] = test_recall
tracking_dict['f1_score'] = test_f1
tracking_dict['auroc'] = test_roc_auc
tracking_dict['loss'] = val_loss[-1]

print('Test AUROC: {}'.format(test_roc_auc))

tracker.record_experiment(tracking_dict)
