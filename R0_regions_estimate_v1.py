#!/usr/bin/env python
# coding: utf-8

# # Инициализация:

# In[1]:


import numpy as np
import pandas as pd
import datetime
import requests
import io
from scipy import stats


# # Глобальные параметры

# In[2]:


## model params
death_shift_days = -21 # нужно будет поменять на 21
cases_shift_days = -11
lethality = 0.0066
rolling_window_for_coeff = 5
tau_e=5.1
tau_i=2.83


# # Функции

# In[3]:


#### Темп удвоения числа заболевших в расчете на каждую дату.
def get_Td(x):
    return np.log(2)/(np.log(1+x))
#### Rₒ
def get_R_naught(Td, tau_e=5.1, tau_i=2.83): # KWARGS
    a = 1 + (tau_e/Td)*np.log(2)
    b = 1 + (tau_i/Td)*np.log(2)
    R_naught = a*b
    return R_naught
def input_coeffs_without_fitting(Series, rolling_window_for_coeff=rolling_window_for_coeff):    
    if Series.dropna().index[-1]==Series.index[-1]:
        return Series
    else:
        Series = Series.copy()
        last_actual_date = Series.dropna().tail().index[-1]
        Series[last_actual_date:][1:] = Series.dropna().tail(rolling_window_for_coeff).mean()
        return Series

def get_tail_trend_slope(Series, tail=rolling_window_for_coeff):
    Series = Series.dropna().tail(tail).reset_index(drop=True).copy()
    xdata = Series.index
    ydata = Series.values
    slope, intercept, r_value, p_value, std_err = stats.linregress(xdata,ydata)
    return slope

def yandex_data_to_convention(df, region='Москва'):
    ndf = df[df['region_name']==region][['cases_delta','deaths_delta']]            .rename(columns={'deaths_delta':'death','cases_delta':'cases'})
    ndf.index.name = 'time'
    return ndf


# In[4]:


def get_R0_Series_for_region(df,                            death_shift_days = -21,                            cases_shift_days = -11,                            lethality = 0.0066,                            rolling_window_for_coeff = 5,                            tau_e=5.1,                            tau_i=2.83,                           ):

    # checking data and params
    ch1 = death_shift_days < cases_shift_days < 0
    ch2 = df[df['death'].cumsum()>0].shape[0] > abs(death_shift_days) # checking that death data is enough
    if not sum([ch1, ch2])==len([ch1, ch2]):
        return -1

    ## Делаем сдвиги рядов данных

    df['death_shifted']  = df['death'].shift(death_shift_days)
    df['cases_shifted']  = df['cases'].shift(cases_shift_days)
    df['cases_expected'] = df['death_shifted']                                .rolling(rolling_window_for_coeff, center=True).mean()                                *(1/lethality)
    df['coeff_cases']    = df['cases_shifted'].rolling(rolling_window_for_coeff, center=True).mean()                            /df['cases_expected']
    df['coeff_cases_forwardshifted'] = df['coeff_cases'].shift(-cases_shift_days)
    df['coeff_cases_model'] = input_coeffs_without_fitting(df['coeff_cases_forwardshifted'])

    ### Считаем время удвоения и Rₒ

    df['cases_adj'] = df['cases']/df['coeff_cases_model']
    df['cases_adj_rate'] = df['cases_adj']/(df['cases_adj'].cumsum())
    df['doubling_time'] = df['cases_adj_rate'].apply(get_Td)
    df['doubling_time_rolling'] = df['doubling_time'].rolling(rolling_window_for_coeff, center=True).mean()
    df['R_naught']=df['doubling_time_rolling'].apply(get_R_naught)
        
    if len(df['R_naught'].dropna())>3:
        return df['R_naught']
    else:
        return -1


# In[5]:


def get_latest_R0_for_region(df,                            death_shift_days = -21,                            cases_shift_days = -11,                            lethality = 0.0066,                            rolling_window_for_coeff = 5,                            tau_e=5.1,                            tau_i=2.83,                           ):

    # checking data and params
    ch1 = death_shift_days < cases_shift_days < 0
    ch2 = df[df['death'].cumsum()>0].shape[0] > abs(death_shift_days) # checking that death data is enough
    if not sum([ch1, ch2])==len([ch1, ch2]):
        return None

    ## Делаем сдвиги рядов данных

    df['death_shifted']  = df['death'].shift(death_shift_days)
    df['cases_shifted']  = df['cases'].shift(cases_shift_days)
    df['cases_expected'] = df['death_shifted']                                .rolling(rolling_window_for_coeff, center=True).mean()                                *(1/lethality)
    df['coeff_cases']    = df['cases_shifted'].rolling(rolling_window_for_coeff, center=True).mean()                            /df['cases_expected']
    df['coeff_cases_forwardshifted'] = df['coeff_cases'].shift(-cases_shift_days)
    df['coeff_cases_model'] = input_coeffs_without_fitting(df['coeff_cases_forwardshifted'])

    ### Считаем время удвоения и Rₒ

    df['cases_adj'] = df['cases']/df['coeff_cases_model']
    df['cases_adj_rate'] = df['cases_adj']/(df['cases_adj'].cumsum())
    df['doubling_time'] = df['cases_adj_rate'].apply(get_Td)
    df['doubling_time_rolling'] = df['doubling_time'].rolling(rolling_window_for_coeff, center=True).mean()
    df['R_naught']=df['doubling_time_rolling'].apply(get_R_naught)
        
    if len(df['R_naught'].dropna())>3:
        return df['R_naught'].dropna()[-1]
    else:
        return None


# In[6]:


def get_R0_trend_for_region(df,                            death_shift_days = -21,                            cases_shift_days = -11,                            lethality = 0.0066,                            rolling_window_for_coeff = 5,                            tau_e=5.1,                            tau_i=2.83,                           ):

    # checking data and params
    ch1 = death_shift_days < cases_shift_days < 0
    ch2 = df[df['death'].cumsum()>0].shape[0] > abs(death_shift_days) # checking that death data is enough
    if not sum([ch1, ch2])==len([ch1, ch2]):
        return None

    ## Делаем сдвиги рядов данных

    df['death_shifted']  = df['death'].shift(death_shift_days)
    df['cases_shifted']  = df['cases'].shift(cases_shift_days)
    df['cases_expected'] = df['death_shifted']                                .rolling(rolling_window_for_coeff, center=True).mean()                                *(1/lethality)
    df['coeff_cases']    = df['cases_shifted'].rolling(rolling_window_for_coeff, center=True).mean()                            /df['cases_expected']
    df['coeff_cases_forwardshifted'] = df['coeff_cases'].shift(-cases_shift_days)
    df['coeff_cases_model'] = input_coeffs_without_fitting(df['coeff_cases_forwardshifted'])

    ### Считаем время удвоения и Rₒ

    df['cases_adj'] = df['cases']/df['coeff_cases_model']
    df['cases_adj_rate'] = df['cases_adj']/(df['cases_adj'].cumsum())
    df['doubling_time'] = df['cases_adj_rate'].apply(get_Td)
    df['doubling_time_rolling'] = df['doubling_time'].rolling(rolling_window_for_coeff, center=True).mean()
    df['R_naught']=df['doubling_time_rolling'].apply(get_R_naught)
    
    Series = df['R_naught'].dropna()

    if len(Series) > 3:
        return get_tail_trend_slope(Series)
    else:
        return None


# # Загружаем последние данные Яндекса

# In[7]:


yandex_covid_data_url = 'https://yastat.net/s3/milab/2020/covid19-stat/data/export/russia_stat.csv'
url=yandex_covid_data_url
s=requests.get(url).content
df=pd.read_csv(io.StringIO(s.decode('utf-8')))
df.to_csv('russia_stat.csv', index=False)


# In[8]:


df = pd.read_csv('russia_stat.csv')
df = df.set_index('date', drop=True)
df.index = pd.to_datetime(df.index)


# # Считаем и экспортируем

# In[9]:


all_regions = df.groupby('region_name').max().sort_values('region_population', ascending=False).index

d1 = {}
d2 = {}
for region in all_regions:
    region_data = yandex_data_to_convention(df,region)
    current_R0 = get_latest_R0_for_region(region_data)
    d1.update({region:    current_R0   })
    if current_R0 != None:
        current_trend_for_region = get_R0_trend_for_region(region_data)
    else:
        current_trend_for_region = None
    d2.update({region: current_trend_for_region})


# In[10]:


df_export = pd.concat([pd.Series(d1).rename('R_naught'),pd.Series(d2).rename('R_naught_trend')], axis = 1)
df_export.index = df_export.index.rename('region_name')
df_export.to_csv('R_naugth_regions.csv')

