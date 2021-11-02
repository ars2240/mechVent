import pandas as pd
import os
import numpy as np
import pickle
import multiprocessing as mp
import math
import random
import feather
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch

from clean_data_laurie import clean_ts, clean_intub
# function defs

def load_data(timeseries_input_directory, static_input_directory):
    '''
    timeseries_input_directory = relative path to directory where all the time series (rrvi, etc.) data files are saved
    Ex: '../data/imputed/'

    static_input_directory = relative path to the directory where the static (intub) data is saved
    static_input_directory = "../data/" '''

    # load initial data
    rrvi = feather.read_dataframe(timeseries_input_directory + 'rrvi.file')
    hrvi = feather.read_dataframe(timeseries_input_directory + 'hrvi.file')
    rr_mean = feather.read_dataframe(timeseries_input_directory + 'rr_mean.file')
    hr_mean = feather.read_dataframe(timeseries_input_directory + 'hr_mean.file')
    sat_mean = feather.read_dataframe(timeseries_input_directory + 'sat_mean.file')

    intub = pd.read_csv(static_input_directory + 'picu_intub.csv')
    intub = clean_intub(intub)
    
    return rrvi, hrvi, rr_mean, hr_mean, sat_mean, intub


# nas will only be the first x missing observations, based on LOCF imputation from imputation nb
def filter_na(key, df):
    if key != 'sat_mean':
        df = df[~df[key+'_norm'].isna()]
    else: 
        df = df[~df[key].isna()]
    return df

# merging pred dfs together
def renamer(df):
    '''
    df -> df
    input: timeseries data recordings
    output: ^
    
    Renames the custom minute column to enable merging on composite PK 
    of encounter ID and minute of recording
    '''
    col = [col for col in df.columns if 'minute' in col][0]
    df = df.rename(columns={col:'minute'}).set_index(['EncID', 'minute'])
    return df

def sampler(df, window_hrs, seed):
    '''
    df -> df
    input: df, random seed val
    output: data filtered such that each encounter has window_hrs * 12 observations (12 observations/hr with
            recordings every 5 min)
    '''
    random.seed(seed)
    
    # subtracting 5 bc 5 minute intervals and including the start point in the count
    window_min = (window_hrs * 60) - 5
    # Jieda note: changed window_min to be 5 minute longer
    # Laurie 11/9: undo this change - it causes irregularly shaped data
    # window_min = (window_hrs * 60)
    # shift minutes by min to make the start 0, recentering
    df['minute_min'] = df.groupby('EncID').minute.transform('min')
    df['minute_shift'] = df['minute'] - df['minute_min']
    df = df.drop('minute_min', axis=1)
    
    # max timestep for each encounter, based on recentered minute
    df_summary = df.groupby('EncID').minute_shift.aggregate(['max'])
    # calculating maximum timestep at which the training window could be started
    # recordings every five min
    df_summary['max_minus_train_hr'] = df_summary['max'] - window_min
    
    def sample_start(max_minus_train_hr):
        '''
        int -> int
        randomly sample a start point for training window
            if the data only contains enough for one sample of train window size
            then sample whole data
        '''
        try:
            return random.sample(range(0, max_minus_train_hr+1, 5), 1)[0]
        
        except ValueError:
            return 0
        
    df_summary['start'] = df_summary.apply(lambda x: sample_start(x['max_minus_train_hr']), axis=1)
    df_summary = df_summary.reset_index()
    
    # rejoining data with sample start
    
    df = df.merge(df_summary, how='inner', on='EncID')
    
    # filtering for obs that are in the correct range
    df = df[(df.minute_shift >= df.start) & (df.minute_shift <= df.start + window_min)]
    return df

def floor_nan(x):
    '''Helper function to apply math.floor in a dataframe with null values.'''
    if not math.isnan(x):
        return math.floor(x)

def make_pred_df(rrvi, rr_mean, hrvi, hr_mean, sat_mean, intub):
    '''Make a prediction dataframe that combines the different time series datasets.
    Inputs are dataframes of rrvi, rr_mean, hrvi, hr_mean, sat_mean time series data.
    
    Also combines the time series data with static predictor data from picu_intub
    
    '''
    # get predictors in one df
    pred_dict = {'rrvi': rrvi,
                'rr_mean': rr_mean,
                'hrvi': hrvi,
                'hr_mean': hr_mean,
                'sat_mean': sat_mean}

    pred_dict = {key: filter_na(key, df) for key, df in pred_dict.items()}
    pred_dict = {key: renamer(df) for key, df in pred_dict.items()}
    pred_df = pd.concat(pred_dict.values(), join='inner', axis=1)

    # creating the full dataset with labels
    pred_df = pred_df.reset_index().merge(intub, how='inner', on='EncID')
    return pred_df

def sample_data(pred_df, pred_hrs, train_hrs, hrs_before_intub, seed, smote=False):
    '''Sample pred_hrs worth of data from both the positive and negative classes.
    
    Train_hrs amount of variation is available in the intubated patients. 
    Only samples up to hrs_before_intub are included for intubated patients.
    If oversample_prop is specified, positive classes will be oversampled to the specified proportion.
    Seed is used as the random ssed in the sampler.'''
    
    pos_samples = pred_df[pred_df['intub_in_picu'] == 1].copy()
    neg_samples = pred_df[pred_df['intub_in_picu'] == 0].copy()
    # filtering positives - this drops positive cases with insufficient data
    pos_samples_full, pos_exc = filter_positives(pos_samples, train_hrs, pred_hrs, hrs_before_intub, smote)
    # filtering negatives - drop negative cases with insufficient data
    neg_samples, neg_exc = filter_negatives(neg_samples, train_hrs, pred_hrs, smote)
    # now take the random samples
    pos_samples = sampler(pos_samples_full, pred_hrs, seed)
    neg_samples = sampler(neg_samples, pred_hrs, seed)
    # reset indices and sort in chronological order within each encounter
    pos_samples = pos_samples.set_index(['EncID', 'minute']).sort_index().reset_index()
    neg_samples = neg_samples.set_index(['EncID', 'minute']).sort_index().reset_index()
    # combine positive and negative samples
    full_data = pos_samples.append(neg_samples)
    return full_data, pos_exc, neg_exc


def fit_score_LR(pred_df, params, file, outcomes_df):
    '''Fit a logistic regression model with the specified parameters and write the outcomes to file.'''
    full_data, pos_exc, neg_exc = sample_data(pred_df, params['pred_hrs'], params['train_hrs'], params['hrs_before_intub'], 
                            params['seed'], params['smote'])
    logistic_data = make_logistic_data(full_data, params)
    
    X = logistic_data.drop(columns = ['EncID', 'intub_minute_cat'])
    # fillna with -1 to represent non-intubated patients
    y = logistic_data['intub_minute_cat'].cat.add_categories(-1).fillna(-1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params['test_size'], 
                                                        random_state=params['seed'])
    lr_mod = LogisticRegression(max_iter=params['max_iter'], 
                                class_weight = params['class_weight']).fit(X_train, y_train)
    
    y_pred_train = lr_mod.predict(X_train)
    y_pred_test = lr_mod.predict(X_test)
    
    # confusion matrix
    train_conf_mat = confusion_matrix(y_train, y_pred_train)
    test_conf_mat = confusion_matrix(y_test, y_pred_test)
#     print("Train confusion matrix:")
#     print(train_conf_mat)
#     print("Test confusion matrix")
#     print(test_conf_mat)
    
    # compute precision, recall, f1-score and micro-f1 score for each of the class
    train_report_dict = classification_report(y_train, y_pred_train, output_dict = True)
    test_report_dict = classification_report(y_test, y_pred_test, output_dict = True)
    
    # record outcomes in a dataframe - easier to sort/filter/compare than printed output
    outcomes = params.copy()
    outcomes['pos_exc'] = pos_exc
    outcomes['neg_exc'] = neg_exc
    # https://www.geeksforgeeks.org/python-program-to-convert-a-list-to-string/
    outcomes['static_preds'] = " ".join(params['static_preds'])
    
    num_cat = len(y_train.unique())
    # get specific scores - we only want the first part of the dictionary because there's extra info at the end
    for label in list(train_report_dict.keys())[:num_cat]:
        outcomes[label+"_precision"] = train_report_dict[label]['precision']
        outcomes[label+"_recall"] = train_report_dict[label]['recall']
        outcomes[label+"_f1-score"] = train_report_dict[label]['f1-score']
        outcomes[label+"_support"] = train_report_dict[label]['support']
    
    # https://stackoverflow.com/questions/17839973/constructing-pandas-dataframe-from-values-in-variables-gives-valueerror-if-usi
    outcomes_df = outcomes_df.append(pd.DataFrame(outcomes, index = [0]))
    
    with open(file, 'a') as f:
        f.write('Run with parameters:' + '\n')
        f.write(str(params) + '\n')
        f.write(str(pos_exc) + " positive samples dropped for insufficient data \n")
        f.write(str(neg_exc) + " negative samples dropped for insufficient data \n")
        f.write("Train confusion matrix: \n" + str(train_conf_mat) + '\n')
        f.write("Test confusion matrix: \n" + str(test_conf_mat) + '\n')
        f.write("Train results: \n" + classification_report(y_train, y_pred_train) + '\n')
        f.write("Test results: \n" + classification_report(y_test, y_pred_test) + '\n')  
        f.write("\n --------------------------------------------------- \n \n")
    
    return outcomes_df

# Jieda note: Here are the functions I made some modifications

def filter_positives(pos_df, train_hrs, pred_hrs, hrs_pre_intub, smote=False):
    
    # Jieda note: changed the full window to (train_hrs +pred_hrs+hrs_pre_intub) *60
 
    # coverting to minutes
    full_window = (train_hrs + pred_hrs+hrs_pre_intub) * 60
    min_pre_intub = hrs_pre_intub * 60
    
    def myround(x, base=5):
        #return int(base * math.ceil(float(x)/base))
        return int(base * math.floor(float(x)/base))
    # filter for greater than 30 min and less than eq to specified hours
    
    # Due to 5 minute increment, and the "intub_minute" being 1-min precise.
    # there is chance that the gap reserved for "hrs_pre_intub" be greater than 60 minutes (61~64 mins)
    # for the sole purpose of truncating, we round intub_minute to nearest 5
    
    pos_df['intub_min_round']=pos_df['intub_minute']
    pos_df['intub_min_round']=pos_df['intub_min_round'].apply(lambda x: myround(x,base=5))
    
    # Laurie 10/31: change the below so that we have hard cutoffs -- absolutely nothing within hrs_before_intub can sneak in
    pos_df = pos_df[(pos_df.intub_min_round - pos_df.minute >= min_pre_intub) & 
                    (pos_df.intub_minute - pos_df.minute < full_window)]  
    # full_window length
    cts = pos_df.groupby(['EncID']).minute.count()
    # recording every 5 min, 12 / hr, need pred hr window
    # Jieda changed 
    if smote:
        recs = (pred_hrs+train_hrs)*12
    else:
        recs = (pred_hrs)*12
    # recs = pred_hrs * 12
    too_few = cts[cts < recs].index
    # filtering out obs with less than 8 hrs worth of recordings
    pos_df = pos_df[~pos_df.EncID.isin(too_few)]
    
    # remove the column 'intub_min_round'
    pos_df.drop(['intub_min_round'],axis=1,inplace=True)
    #

    return pos_df, len(too_few)


def filter_negatives(neg_df, train_hrs, pred_hrs, smote=False):
    '''
    df -> df
    input: df of non-intubated patients, size of training window in hours
    output: df filtered for obs containing enough recordings to construct a row of training data for each enc
            Also returns count of the filtered data
    '''
    # filtering out less than (train_hrs + pred_hrs) of observations
    # Jieda changed recs count
    if smote:
        recs = (pred_hrs+train_hrs)*12
    else:
        recs = (pred_hrs)*12
    cts = neg_df.groupby('EncID').minute.count()
    too_few = cts[cts<recs].index
    neg_df = neg_df[~neg_df.EncID.isin(too_few)]
    return neg_df, len(too_few)


# make logistic data
def make_logistic_data(full_data, params):
    '''Get time series summary statistics, last thirty minutes of pred window summary statistics, 
    static predictors, and an hour of intubation categorical variable.'''

    # time series data
    # prepare to aggregate 
    full_time_data = full_data[['EncID', 'rrvi_norm','rr_mean_norm','hrvi_norm','hr_mean_norm','sat_mean']]
    # run the summary statistics on the entire time window
    full_time_summary = full_time_data.groupby('EncID').agg(['mean', 'median', 'std'])
    # get rid of multiindex
    # https://stackoverflow.com/questions/44023770/pandas-getting-rid-of-the-multiindex
    full_time_summary.columns = full_time_summary.columns.map('_'.join)
    # get EncID out of the index
    full_time_summary.reset_index(inplace = True)
    
    # hour-level categorical variable
    # pull out the relevant columns
    make_categorical = full_data[['EncID', 'minute', 'intub_in_picu', 'intub_minute']].groupby('EncID').max()
    # categorical version of intub_minute comes from getting:
    # intub_minute minus maximum minute present in prediction timeseries minus the number of hours in the hrs_before_intub
    # need to also account for the granularity in pred_gran - either 1 hr units or 30 min
    make_categorical['intub_minute_cat'] = ((make_categorical['intub_minute'] - make_categorical['minute']) - (60 * params['hrs_before_intub'])) / (60 * params['pred_gran'])

    # round to lower integer and cast as category
    make_categorical['intub_minute_cat'] = make_categorical['intub_minute_cat'].apply(lambda x: floor_nan(x))
    
    # cast as category
    make_categorical['intub_minute_cat']=make_categorical['intub_minute_cat'].astype('category')
    
    intub_minute_cat = make_categorical.reset_index()[['EncID', 'intub_minute_cat']]
    
    # last thirty minutes
    last_thirty = full_data[['EncID', 'minute', 'rrvi_norm', 'rr_mean_norm', 'hrvi_norm',
           'hr_mean_norm', 'sat_mean']].groupby('EncID').tail(7)
    last_thirty = last_thirty.groupby('EncID').agg(['mean', 'median', 'std'])
    # get rid of multiindex
    # https://stackoverflow.com/questions/44023770/pandas-getting-rid-of-the-multiindex
    last_thirty.columns = last_thirty.columns.map('_'.join)
    last_thirty.drop(columns = ['minute_mean', 'minute_median', 'minute_std'], inplace= True)
    # get EncID out of the index
    last_thirty.reset_index(inplace = True)
    
    # static predictors
    statics = full_data[['EncID'] + params['static_preds']].drop_duplicates()
    
    logistic_data = statics.merge(full_time_summary, how = "left", 
    on = "EncID").merge(last_thirty, how = "left", on = "EncID", suffixes = ["_full", "_last30"]).merge(intub_minute_cat)
    
    return logistic_data

def create_Xy(df, pred_hrs, time_preds, static_preds):
    '''
    df, int, list -> X, y (array, array)
    input dataframe, number of hours use to train model, list of predictors
    output: X, predictor matrix of shape (n encounters, 12*pred_hrs, len(preds)), y labels of shape n_encounters
    
    '''
    # recording every 5 min, 12 recs/hr
    train_recs = 12 * pred_hrs
    n_encs = df.EncID.nunique()
    y = df.filter(['intub_in_picu'], axis=1).to_numpy().reshape(n_encs, train_recs, 1).mean(axis=1)
    df_time = df.filter(time_preds, axis=1)
    X_time = df_time.to_numpy().reshape(n_encs, train_recs, len(time_preds))
    df_static = df.filter(static_preds + ['EncID'], axis=1).drop_duplicates().filter(static_preds, axis=1)
    X_static = df_static.to_numpy().reshape(n_encs,len(static_preds))
    return X_time, X_static, y


def train_val_test_split(X_time, X_static, y, train_prop: float, val_prop: float, test_prop: float):
    '''
    Input: predictor array, label array, proportions of data for train validatoin and testing
    Output: train, validation, test arrays in form X, y, X, y, X, y
    splits data into training, validation, and test sets
    '''
    assert train_prop + val_prop + test_prop == 1.0, 'Split must partition the data'
    
    n_train = int(train_prop * X_time.shape[0])
    n_val = int(val_prop * X_time.shape[0])
    n_test = int(test_prop * X_time.shape[0])
    available_indices = set(range(X_time.shape[0]))
    
    train_indices = random.sample(list(available_indices), n_train)
    available_indices -= set(train_indices)
    val_indices = random.sample(list(available_indices), n_val)
    test_indices = list(available_indices - set(val_indices))
    
    
    X_time_train = X_time[train_indices, :, :]
    X_stat_train = X_static[train_indices, :]
    y_train = y[train_indices,]
    X_time_val = X_time[val_indices,:, :]
    X_stat_val = X_static[val_indices, :]
    y_val = y[val_indices,]
    X_time_test = X_time[test_indices, :, :]
    X_stat_test = X_static[test_indices, :]
    y_test = y[test_indices,]
    
    return X_time_train, X_stat_train, y_train, X_time_val, X_stat_val, y_val, X_time_test, X_stat_test, y_test
