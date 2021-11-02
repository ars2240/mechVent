import pandas as pd
import pickle
import os
import numpy as np
import math


def mean_sd_ts(intub, ts_df):
    '''
    DataFrame, DataFrame -> DataFrame
    Calculate mean and std of time series values by age
    Must normalize all time series data except o2 sat which is not age dependent.
    '''
    # minute column
    key_min = [col for col in ts_df.columns if 'minute' in col][0]
    # recorded variable/predictor
    key = key_min.split('_minute')[0]
    # filtering out deaths
    intub_nodeath = intub[intub.diedinHosp == 0].EncID
    # selecting timesteps from patients who have not died
    ts_df_nodeath = ts_df[ts_df.EncID.isin(intub_nodeath)].copy()
    # creating max value for minutes column by encounter
    ts_df_nodeath['max'] = ts_df_nodeath.groupby(
        'EncID')[key_min].transform('max')
    # twelve hours in minutes
    hrs = 12*60
    # selecting last 12 hours of recordings before discharge
    # i.e. when the patient is healthiest and ready to come out of PICU
    ts_df_nodeath = ts_df_nodeath[ts_df_nodeath['max'] -
                                  ts_df_nodeath[key_min] <= hrs]
    ts_result = ts_df_nodeath.groupby(
        'age_yr_19ct')[key].aggregate(['mean', 'std', 'count'])
    return ts_result


def clean_ts(ts_data, intub_dat):
    '''
    Function that puts together all cleaning steps for the ts data
    '''
    minute = [col for col in ts_data.columns if 'minute' in col][0]
    clean_ts_data = ts_data[ts_data[minute] >= 0]
    clean_ts_data = clean_ts_data[clean_ts_data.EncID.isin(intub_dat.EncID)]
    try:
        clean_ts_data = clean_ts_data.drop('age_mo', axis=1)
    # one of the dfs doesn't have age_mo as a col
    except KeyError:
        pass
    intub_age = intub_dat.filter(['EncID', 'age_mo'])
    clean_ts_data = pd.merge(clean_ts_data, intub_age, how='inner', on='EncID')
    clean_ts_data['age_yr'] = clean_ts_data.age_mo // 12
    clean_ts_data['age_yr_19ct'] = [yr if yr < 18 else 18
                                    for yr in clean_ts_data.age_yr]
    # creating a normalized column value
    v_name = minute.split('_minute')[0]
    # sat_mean is not age dependent
    if v_name != 'sat_mean':
        ts_stat = mean_sd_ts(intub_dat, clean_ts_data)
        clean_ts_data = pd.merge(clean_ts_data, ts_stat, left_on='age_yr_19ct',
                                 right_on=ts_stat.index)
        clean_ts_data[v_name+'_norm'] = (clean_ts_data[v_name] -
                                         clean_ts_data['mean'])/clean_ts_data['std']

    return clean_ts_data


def clean_intub(intub_dat):
    '''
    Function that puts together all cleaning steps for intub data
    '''
    with open('../bw_looking_missing.pkl', 'rb') as f:
        missing_summary = pickle.load(f)

    missing = missing_summary.query('rrvi_minute_prop >= 0.5 or\
                                    rr_mean_minute_prop >= 0.5 or\
                                    hrvi_minute_prop >= 0.5 \
                                    or hr_mean_minute_prop >= 0.5 or\
                                    sat_mean_minute_prop >= 0.5')
    missing_encs = missing.index.to_list()
    intub_dat = intub_dat[~intub_dat.EncID.isin(missing_encs)].copy()
    # putting weekday range between 0 and 6. This adis in calculation
    # of intub at off hour column
    intub_dat['weekday_admit'] = intub_dat['weekday_admit'] - 1
    # creating off hour column
    # Off hour helper function

    def off_hr(dow, hr):
        '''
        off hour calculation helper function
        '''
        oh_indicator = 0
        # 7 is saturday, 1 is sunday, 17 is 5 pm, 8 is 8 am
        # hospital off hours are weekends and 5pm-8am
        if dow in [0, 6] or hr >= 17 or hr <= 8:
            oh_indicator = 1
        return oh_indicator

    def off_hr_intub(intub_minute, dow, admit_time):
        # handling non-intubations
        if math.isnan(intub_minute):
            return np.NaN
        # rounding down hours after admission
        hours_aft_admit = math.floor(intub_minute / 60)
        # adding days if necessary
        day_ctr = math.floor(hours_aft_admit/24)
        # counting number of hours mod 24
        add_hour = hours_aft_admit % 24
        # adding the intub_time hour to admit time
        intub_hr = admit_time + add_hour
        # if sum is over 23 (midnight) mod 24 and add a day
        if intub_hr > 23:
            intub_hr = intub_hr % 24
            day_ctr += 1
        # handling weekday
        intub_day = (dow + day_ctr) % 7
        # checking if the day and hour qualify as off hours
        intub_at_off_hr = off_hr(intub_day, intub_hr)
        return intub_at_off_hr

    intub_dat['off_hour'] = intub_dat.apply(lambda x:
                                            off_hr(x['weekday_admit'],
                                                   x['hour_admit']), axis=1)

    intub_dat['off_hour_intub'] = intub_dat.apply(lambda x:
                                                  off_hr_intub(
                                                      x['intub_minute'],
                                                      x['weekday_admit'],
                                                      x['hour_admit']), axis=1)
    intub_dat['age_yr'] = intub_dat.age_mo // 12
    intub_dat['age_yr_19ct'] = [yr if yr <
                                18 else 18 for yr in intub_dat.age_yr]

    return intub_dat
