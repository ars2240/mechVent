import os
import pandas as pd
import pickle
import keras
from keras import layers
import tensorflow as tf
# from experiment_tracking import ExperimentTracker
import csv
# file with useful modeling functions
import lamk_data_prep as ldp
import keras_dict_dataset as kdd
import keras_nnfuncs as nnfuncs
from sklearn.metrics import auc, roc_curve
import csv_tracker as track
from datetime import datetime

# note: at present this script seems to consistently lead to nan loss values during training
# I experimented with options in this thread but did not have much luck: https://stackoverflow.com/questions/37232782/nan-loss-when-training-regression-network 

# user inputs - script is designed so that you can just make changes in this seection and then re-run for testing purposes
# data directory where the time series data will be loaded from
time_data_dir = "../data/imputed/"
# data directory where the static data will be loaded from
static_data_dir = "../data/raw_data/"
# model prefix -- will appear in tracking file and artefact file names
model_prefix = "keras_rsampl_stat_l2"
# whether to re-make the negative and positive sample data or load existing from file
# if changing train, pred, or before_intub hours should NOT use "existing" 
# set to "new" to save out the new positive and negative samples generated in this run
neg_pos_save = "existing"
# same deal as negative/positive data - can set to new to save out or existing to load existing save of train/val/test pickled dictionary data
train_val_test_save = "existing"
# flag whether to use resampling during training
resample_flag = True
# whether to use static predictors - this script requires True
stat = True
# whether to shuffle data in training
shuffle = True
# number of hours for the prediction period (yes, it's confusingly named)
train_hrs = 4
# number of hours for the data collection period
pred_hrs = 4
# gap between end of data collection window and first possible intubation time
hrs_before_intub = .5
# random seed to use in places where a seed is possible
seed = 117
# proportion of data to use for training, validation, and test (must sum to 1)
train_prop = .6
val_prop = .2
test_prop = .2
# batch size to use in nn training
batch_size = 128
# list of column names for time series and static  predictors
time_preds = ['rrvi_norm', 'rr_mean_norm', 'hrvi_norm', 'hr_mean_norm', 'sat_mean']
static_preds = ['malignancy', 'transpl', 'sum_comorb_ccc']
# model architecture
# num obs in training sequence - should equal pred_hrs * 12 (12 = 60/5 min)
sequence_length = 48
# number of nodes in hidden layers in network
hidden_size = 128
# learning rate
lr = 0.0001
# training epochs
num_epochs = 10

print("loading data at " + str(datetime.now()))
if neg_pos_save != "existing":
    # load and reshape data
    # make the full predictor df
    pred_df = kdd.make_pred(time_data_dir, static_data_dir)
    # filter based on the hour limits indicated 
    pos_samples_f, neg_samples_f = kdd.make_filtered_df(pred_df, pred_hrs, train_hrs, hrs_before_intub)
    if neg_pos_save == "new":
        pos_samples_f.to_csv("pos_samples_f.csv", index = False)
        neg_samples_f.to_csv("neg_samples_f.csv", index = False)
else:
    pos_samples_f = pd.read_csv("pos_samples_f.csv")
    neg_samples_f = pd.read_csv("neg_samples_f.csv")

# train test split
print("train val test split at " + str(datetime.now()))
if train_val_test_save != "existing":
    train, val, test = kdd.make_train_val_test(pos_samples_f, neg_samples_f, seed, train_prop, val_prop, test_prop, time_preds, static_preds, pred_hrs, resample_flag)
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
print("build train generator at " + str(datetime.now() ))
train_data = kdd.CustomSequence(train, pred_hrs, stat, batch_size, shuffle)
print("build val generator at " + str(datetime.now()))
val_data = kdd.CustomSequence(val, pred_hrs, stat, batch_size, shuffle)
# combine val and train: https://www.geeksforgeeks.org/python-merging-two-dictionaries/
full_train_data = kdd.CustomSequence({**train, **val}, pred_hrs, stat, batch_size, shuffle)
print("build test generator at " + str(datetime.now()))
test_data = kdd.CustomSequence(train, pred_hrs, stat, batch_size, shuffle)

# epochs, hiddensize, LR
model_id = model_prefix + str(num_epochs) + '_' + str(hidden_size) + \
    '_' + str(lr).split('.')[1]

tracker = track.CSVTracker('../results/experiment_tracking.csv')

tracker.is_model_unique(model_id)

# creating a model_id folder in which plots can be saved
if not os.path.isdir('results/' + model_id):
    os.mkdir('results/' + model_id)

# start model definition
# follow: https://keras.io/guides/functional_api/#models-with-multiple-inputs-and-outputs
# see also: https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
# and : https://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/
time_input = keras.Input(shape = (sequence_length, len(time_preds)), name = "time")
time_feat = layers.LSTM(hidden_size)(time_input)
if stat:
    stat_input = keras.Input(shape = (1,len(static_preds)), name = "static")
    stat_feat = layers.Dense(1)(stat_input)
    # https://stackoverflow.com/questions/53363986/adding-static-data-not-changing-over-time-to-sequence-data-in-lstm/53373263#53373263
    # https://stackoverflow.com/questions/52579936/concatenate-additional-features-after-lstm-layer-for-time-series-forecasting
    stat_feat = layers.Flatten()(stat_feat)
    x = layers.concatenate([time_feat, stat_feat])
    # one output node: https://stats.stackexchange.com/questions/207049/neural-network-for-binary-classification-use-1-or-2-output-neurons
    output = layers.Dense(1, activation = "softmax")(x)
    model = keras.Model(inputs = [time_input, stat_input], outputs = output)
else:
    output = layers.Dense(1, activation = "softmax")(time_feat)
    model = keras.Model(inputs = time_input, outputs = output)

opt = keras.optimizers.Adam(lr=lr, clipvalue=1)
model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = [keras.metrics.Accuracy(), keras.metrics.AUC(),keras.metrics.Precision(), keras.metrics.Recall()])
history = model.fit_generator(train_data, validation_data = val_data, epochs = num_epochs)

# nnfuncs.plot_loss(model_id, train_loss, val_loss)
model.save('results/' + model_id + '/model.h5')
# test results
results = model.evaluate(test_data)
test_acc = results[1]
test_precision = results[3]
test_recall = results[4]
test_f1 = (2 * test_precision * test_recall) / (test_precision + test_recall)
test_roc_auc = results[2]

# test_acc = nnfuncs.model_accuracy(test_pred, test_label, 0.5)
# test_precision, test_recall, test_f1 = nnfuncs.precision_recall_f1(test_label, test_pred, 0.5)


# pr_path = 'results/' + model_id + '/' + model_id + '_precision_recall.png'
# nnfuncs.plot_test_precision_recall(pr_path, test_label, test_pred)


# plotting and saving ROC curve as well
# test_fpr, test_tpr, _ = roc_curve(test_label.cpu(), test_pred.cpu())
# test_roc_auc = auc(test_fpr, test_tpr)

# test_roc_path = 'results/' + model_id + '/' + model_id + '_test_roc.png'
# nnfuncs.plot_roc(test_fpr, test_tpr, test_roc_auc, test_roc_path)

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
tracking_dict['loss'] = results[0]

print('Test AUROC: {}'.format(test_roc_auc))

tracker.record_experiment(tracking_dict)
