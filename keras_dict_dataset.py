import pickle
import torch
import keras
import pandas as pd # need version >= 1.1.0 to use groupby shuffle
import random
import numpy as np
import lamk_data_prep as dp


'''
This module provides the necessary functions and implementation of the datset
class to handle intubation data set up as a dictionary with encounter id as
the key and a dictionary of data, y as the value

Functions:
    filter_dict:
        Filters the dictionary so that only encounters with specified amt of
        data are available. Takes a random sample of size
        12*train_hrs (measurements every 5 min so 12 recordings/hr).
        For intubated patients records the number of minutes between
        last obs and intubation time. Saves with key 'last_rec'


Classes:
    Dataset class used to construct train-val-test datasets

'''


def make_pred(time_series_directory, static_directory):
    # load data
    rrvi, hrvi, rr_mean, hr_mean, sat_mean, intub = dp.load_data(time_series_directory, static_directory)
    # combine into full predictor dataframe
    pred_df = dp.make_pred_df(rrvi, rr_mean, hrvi, hr_mean, sat_mean, intub)
    return pred_df

def make_filtered_df(pred_df, pred_hrs, train_hrs, hrs_before_intub, smote=False):
    '''Filter out rows with insufficient data. Return positive and negative data separately so they can be sampled.'''
    
    # first: filter out cases with insufficient data
    pos_samples = pred_df[pred_df['intub_in_picu'] == 1].copy()
    neg_samples = pred_df[pred_df['intub_in_picu'] == 0].copy()
    # filtering positives - this drops positive cases with insufficient data
    pos_samples_f, pos_exc = dp.filter_positives(pos_samples, train_hrs, pred_hrs, hrs_before_intub, smote)
    # filtering negatives - drop negative cases with insufficient data
    neg_samples_f, neg_exc = dp.filter_negatives(neg_samples, train_hrs, pred_hrs, smote)
    
    # reset indices and sort in chronological order within each encounter
    pos_samples_f = pos_samples_f.set_index(['EncID', 'minute']).sort_index().reset_index()
    neg_samples_f = neg_samples_f.set_index(['EncID', 'minute']).sort_index().reset_index()
    
    return pos_samples_f, neg_samples_f

def train_val_test_split(df, seed, train_prop, val_prop, test_prop):
    '''Split dataframe into train, val, test, of specified proportions (_prop) using seed as random state.'''
    assert train_prop + val_prop + test_prop == 1.0, 'Split must partition the data'
    
    # sample from the positives
    # get number of unique encounters
    num = len(df['EncID'].unique())
    # get numbers for each set
    n_train = int(train_prop * num)
    n_val = int(val_prop * num)
    n_test = int(test_prop * num)
    # filter the list of encounters to the appropriate numbers
    # https://stackoverflow.com/questions/24147278/how-do-i-create-test-and-train-samples-from-one-dataframe-with-pandas
    encs = pd.DataFrame(df['EncID'].unique())
    train = encs.sample(n = n_train, random_state = seed)
    remaining = encs[~encs.isin(train)].dropna()
    val = remaining.sample(n = n_val, random_state = seed)
    test = remaining[~remaining.isin(val)].dropna()
    
    train_df = df[df['EncID'].isin(train[0])]
    val_df = df[df['EncID'].isin(val[0])]
    test_df = df[df['EncID'].isin(test[0])]
    # this is kind of hacky but to normalize the sum_comorb_ccc predictor, we only want to use the values based on the training set so just do it now
    if 'sum_comorb_ccc' in list(train_df.columns): 
        maxc = max(train_df['sum_comorb_ccc'])
        minc = min(train_df['sum_comorb_ccc'])
        for df in [train_df, val_df, test_df]:
             df['sum_comorb_ccc'] = (df['sum_comorb_ccc'].copy() - minc) / maxc    
    return train_df, val_df, test_df

def make_train_val_test(pos_df, neg_df, seed, train_prop, val_prop, test_prop, time_preds, static_preds, pred_hrs, resample_flag = True):
    '''Take positive and negative dataframes and apply train, val, test split'''
    
    train_df = pd.DataFrame()
    val_df = pd.DataFrame()
    test_df = pd.DataFrame()
    
    # shuffle order, following: https://stackoverflow.com/questions/45585860/shuffle-a-pandas-dataframe-by-groups
    df_list = [train_df, val_df, test_df]
    shuffled_list = []
    for i in range(len(df_list)):
        df = df_list[i]
        groups = [data for _, data in df.groupby('EncID')]
        random.seed(seed)
        random.shuffle(groups)
        shuffled_list.append(pd.concat(groups).reset_index(drop=True))
    
    train_df = shuffled_list[0]
    val_df = shuffled_list[1]
    test_df = shuffled_list[2]
    
    # convert to dict at this point
    # check resample flag for whether we want to return the whole train/val sets or set a fixed sample at this point
    if resample_flag:  
        train = df_to_dict(train_df,time_preds, static_preds)
        val = df_to_dict(val_df, time_preds, static_preds)
    else:
    # fix the datasets - performance should be similar to baseline
        train = fix_test(df_to_dict(train_df,time_preds, static_preds),pred_hrs)
        val = fix_test(df_to_dict(val_df, time_preds, static_preds),pred_hrs)
    
    test = fix_test(df_to_dict(test_df, time_preds, static_preds), pred_hrs)
    
    # for debugging
    print("train: " + str(len(train)))
    print("val: " + str(len(val)))
    print("test: " + str(len(test)))
    return train, val, test 

def df_to_dict(df, time_preds, static_preds):
    '''
    DataFrame -> dictionary (of DataFrames/dictionaries/attributes)
    input:
        dataframe with predictors, response, intub_minute, minute columns
        time_preds input should be a list *that includes minute* of the time series predictors
        static_preds are the static predictors
    
    output:
        dictionary with encounter id as key and predictors, response, etc as key/value pairs
    '''
    grouped_df = df.groupby('EncID')
    df_dict = {}
    for group in grouped_df.groups:
        df_dict[group] = {}
        # data will contain time series preds
        df_dict[group]["data"] = grouped_df.get_group(group).drop('EncID', axis=1).filter(time_preds)
        # static contains static preds
        df_dict[group]["static"] = grouped_df.get_group(group).drop('EncID', axis=1).filter(static_preds).drop_duplicates().to_numpy()
        # next commented line is a version of the static which returns labeled data instead of a numpy array
        # df_dict[group]["static"] = grouped_df.get_group(group).drop('EncID', axis=1).filter(static_preds).drop_duplicates().to_dict(orient = "records")
        # y contains labels 
        df_dict[group]["y"] = grouped_df.get_group(group).drop('EncID', axis=1).filter(['intub_in_picu'])['intub_in_picu'].unique()[0]
    
    return df_dict
    
def fix_test(test, pred_hrs):
    '''Fix the test sample so it can't be resampled. Applied after test set is converted to dict.'''
    for enc in test.keys():
        num_steps = int(12 * pred_hrs)
        last_valid = len(test[enc]['data']) - num_steps
        start = random.randint(0, last_valid)
        end = start + num_steps
        test[enc]['data'] = test[enc]['data'].iloc[start:end, :]
    return test
    
def df_to_tensor(data_dict: dict, preds: list):
    for value in data_dict.values():
        pred_df = value['data'].filter(preds, axis=1)
        pred_array = pred_df.to_numpy()
        pred_tensor = torch.from_numpy(pred_array)
        value['data'] = pred_tensor
        
def build_dataset(dict_dataset, pred_hrs: float, batch_size: int, static: bool):
    tensordata = CustomDataset(dict_dataset, pred_hrs, static)
    dataloader = DataLoader(tensordata, batch_size=batch_size, shuffle=True)
    return dataloader

# follow: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly#conclusion
# uses MIT license
class CustomSequence(keras.utils.Sequence):
    def __init__(self, data_dict: dict, pred_hrs: float, stat: bool, batch_size: int, shuffle = True):
        self.encounters = list(data_dict.keys())
        self.num_time_preds = len(list(data_dict.values())[0]['data'].columns)
        self.num_stat_preds = list(data_dict.values())[0]['static'].shape[1]
        self.data_dict = data_dict
        self.pred_hrs = pred_hrs
        self.num_steps = int(12 * pred_hrs)
        self.shuffle = shuffle
        self.stat = stat
        self.batch_size = batch_size
        self.on_epoch_end()

    def on_epoch_end(self):
        '''Updates indexes after each epoch - from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly#conclusion'''
        self.indexes = np.arange(len(self.encounters))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        '''num encounters per batch'''
        return int(np.floor(len(self.encounters) / self.batch_size))

    def __getitem__(self, idx):
        # NOTE: for keras/TF you need to return a BATCH here rather than an individual sample as in torch
        # the index here is a BATCH index not an encounter index

        # Generate indexes of the batch
        idx_list = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]

        # Generate data
        X, y = self.__batch_generation(idx_list)

        return X, y
        
    def __batch_generation(self, batch_IDs):
        '''batch_IDs is a list of indices which need to be looked up in the list of encounters
           returns a batch of X and y values based on whether we are or are not using static preds'''
        for lidx, enc in enumerate(batch_IDs):
            if not self.stat:                
                X = np.empty((self.batch_size, self.num_steps, self.num_time_preds))
                y = np.empty((self.batch_size), dtype=int)
                time, label = self.__sample_generation(enc)
                X[lidx,] = time
                y[lidx] = label
                # return X, keras.utils.to_categorical(y, num_classes=1)
                return X, y
            else: 
                X_time = np.empty((self.batch_size, self.num_steps, self.num_time_preds))
                X_stat = np.empty((self.batch_size, 1, self.num_stat_preds))
                y = np.empty((self.batch_size), dtype=int)
                time, stat, label = self.__sample_generation(lidx)
                y[lidx] = label
                X_time[lidx,] = time
                X_stat[lidx,] = stat
                # to reshape multiple inputs 
                # https://stackoverflow.com/questions/57512596/how-to-generate-sequence-data-with-keras-with-multiple-input
                # https://stackoverflow.com/questions/55886074/keras-fit-generator-with-multiple-input-layers
                # return [X_time, X_stat], [keras.utils.to_categorical(y, num_classes=1)]
                return [X_time, X_stat], [y]

    def __sample_generation(self, idx):
        # this method samples a window of
        # size num_steps from a sample
        ID = self.encounters[idx]
        # if we are using static predictors, return X_time, X_stat, and y; else just return X_time and y
        if not self.stat:
            X, y = self.data_dict[ID]['data'], self.data_dict[ID]['y']
            X = self._sampler(X)
            X_array = X.to_numpy()
            # X_tensor = torch.from_numpy(X_array)
            return X_array, y
        else: 
            X_time, X_stat, y = self.data_dict[ID]['data'], self.data_dict[ID]['static'], self.data_dict[ID]['y']
            X_time = self._sampler(X_time)
            X_time_array = X_time.to_numpy()
            # X_time_tensor = torch.from_numpy(X_time_array)
            # X_stat_tensor = torch.from_numpy(X_stat)
            return X_time_array, X_stat, y

    def _sampler(self, X):
        last_valid = len(X.index) - self.num_steps
        start = random.randint(0, last_valid)
        end = start + self.num_steps
        result = X.iloc[start:end, :]
        return result
