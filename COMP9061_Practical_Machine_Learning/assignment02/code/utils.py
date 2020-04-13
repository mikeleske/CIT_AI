import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

import matplotlib.pyplot as plt
import seaborn as sns
import os
import math

def add_time(df):
    ''' Transform the pickup_datetime information into 'year', 'weekday' and 'hour' columns.
        This is expected to help the algorithms to learn about price increases and rush hour
        and/or night fares.
    '''
    df['year'] = df.pickup_datetime.apply(lambda t: t.year)
    df['weekday'] = df.pickup_datetime.apply(lambda t: t.weekday())
    df['hour'] = df.pickup_datetime.apply(lambda t: t.hour)
    
def distance(lat1, lon1, lat2, lon2):
    ''' Calculate the ride distance based on Haversine distance.
    '''
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))

def add_travel_vector_features(df):
    ''' Add new column to dataframe with distance in miles.
    ''' 
    df['distance_miles'] = distance(df.pickup_latitude, df.pickup_longitude, \
                                          df.dropoff_latitude, df.dropoff_longitude)

def add_airport_dist(df):
    ''' Add minumum distance from pickup or dropoff coordinates to each airport.
        JFK: John F. Kennedy International Airport
        EWR: Newark Liberty International Airport
        LGA: LaGuardia Airport
    '''
    jfk_coord = (40.639722, -73.778889)
    ewr_coord = (40.6925, -74.168611)
    #lga_coord = (40.77725, -73.872611)
    
    pickup_lat = df['pickup_latitude']
    dropoff_lat = df['dropoff_latitude']
    pickup_lon = df['pickup_longitude']
    dropoff_lon = df['dropoff_longitude']
    
    pickup_jfk = distance(pickup_lat, pickup_lon, jfk_coord[0], jfk_coord[1]) 
    dropoff_jfk = distance(jfk_coord[0], jfk_coord[1], dropoff_lat, dropoff_lon) 
    pickup_ewr = distance(pickup_lat, pickup_lon, ewr_coord[0], ewr_coord[1])
    dropoff_ewr = distance(ewr_coord[0], ewr_coord[1], dropoff_lat, dropoff_lon) 
    #pickup_lga = distance(pickup_lat, pickup_lon, lga_coord[0], lga_coord[1]) 
    #dropoff_lga = distance(lga_coord[0], lga_coord[1], dropoff_lat, dropoff_lon) 
    
    df['jfk_dist'] = pd.concat([pickup_jfk, dropoff_jfk], axis=1).min(axis=1)
    df['ewr_dist'] = pd.concat([pickup_ewr, dropoff_ewr], axis=1).min(axis=1)
    #df['lga_dist'] = pd.concat([pickup_lga, dropoff_lga], axis=1).min(axis=1)
    
def add_features(df):
    ''' Bundle fixed feature engineering.
    '''
    add_time(df)
    add_travel_vector_features(df)
    add_airport_dist(df)
    
    return df

def drop_date(df):
    ''' Drop 'pickup_datetime' column from dataframe.
    '''
    df = df.drop(['pickup_datetime'], axis=1)
    return df

def clean_df(df):
    ''' Data cleaning as per description in report.
    '''
    print('Old size: %d' % len(df))
    
    # Remove observations with missing values
    df.dropna(how='any', axis='rows', inplace=True)

    # Removing observations with erroneous values
    mask = df['pickup_longitude'].between(-75, -73)
    mask &= df['dropoff_longitude'].between(-75, -73)
    mask &= df['pickup_latitude'].between(40, 42)
    mask &= df['dropoff_latitude'].between(40, 42)
    mask &= df['passenger_count'].between(0, 6)
    mask &= df['fare_amount'].between(2.5, 500)
    mask &= df['distance_miles'].between(0.05, 200)

    df = df[mask]
    
    print('New size: %d' % len(df))
    
    return df

def scale_gps(df, sc, lat_scaler=None, lon_scaler=None):  
    ''' Scale latitude and longitude columns.
        Return scaler objects so that the same scaler 
        can be used to transform the test data.
    '''
    coords = np.vstack((
        df[['pickup_latitude', 'pickup_longitude']].values,
        df[['dropoff_latitude', 'dropoff_longitude']].values
    ))

    if not lat_scaler:
        if sc == 'MinMaxScaler':
            lat_scaler = MinMaxScaler().fit(coords[:, 0].reshape(-1, 1))
            lon_scaler = MinMaxScaler().fit(coords[:, 1].reshape(-1, 1))
        elif sc == 'minmax_scale':
            lat_scaler = minmax_scale().fit(coords[:, 0].reshape(-1, 1))
            lon_scaler = minmax_scale().fit(coords[:, 1].reshape(-1, 1))
        elif sc == 'MaxAbsScaler':
            lat_scaler = MaxAbsScaler().fit(coords[:, 0].reshape(-1, 1))
            lon_scaler = MaxAbsScaler().fit(coords[:, 1].reshape(-1, 1))
        elif sc == 'StandardScaler':
            lat_scaler = StandardScaler().fit(coords[:, 0].reshape(-1, 1))
            lon_scaler = StandardScaler().fit(coords[:, 1].reshape(-1, 1))
        elif sc == 'RobustScaler':
            lat_scaler = RobustScaler().fit(coords[:, 0].reshape(-1, 1))
            lon_scaler = RobustScaler().fit(coords[:, 1].reshape(-1, 1))
        elif sc == 'Normalizer':
            lat_scaler = Normalizer().fit(coords[:, 0].reshape(-1, 1))
            lon_scaler = Normalizer().fit(coords[:, 1].reshape(-1, 1))
        elif sc == 'QuantileTransformer':
            lat_scaler = QuantileTransformer().fit(coords[:, 0].reshape(-1, 1))
            lon_scaler = QuantileTransformer().fit(coords[:, 1].reshape(-1, 1))
        elif sc == 'PowerTransformer':
            lat_scaler = PowerTransformer().fit(coords[:, 0].reshape(-1, 1))
            lon_scaler = PowerTransformer().fit(coords[:, 1].reshape(-1, 1))

    df[['pickup_latitude']] = pd.DataFrame(lat_scaler.transform(df[['pickup_latitude']]))
    df[['pickup_longitude']] = pd.DataFrame(lon_scaler.transform(df[['pickup_longitude']]))
    df[['dropoff_latitude']] = pd.DataFrame(lat_scaler.transform(df[['dropoff_latitude']]))
    df[['dropoff_longitude']] = pd.DataFrame(lon_scaler.transform(df[['dropoff_longitude']]))
    
    return df, lat_scaler, lon_scaler

def pca_gps(df, pca=None):
    ''' PCA transform latitude and longitude features.
    '''
    coords = np.vstack((
        df[['pickup_latitude', 'pickup_longitude']].values,
        df[['dropoff_latitude', 'dropoff_longitude']].values
    ))
    
    if not pca:
        pca = PCA().fit(coords)
        
    df['pickup_pca0'] = pca.transform(df[['pickup_latitude', 'pickup_longitude']])[:, 0]
    df['pickup_pca1'] = pca.transform(df[['pickup_latitude', 'pickup_longitude']])[:, 1]
    df['dropoff_pca0'] = pca.transform(df[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
    df['dropoff_pca1'] = pca.transform(df[['dropoff_latitude', 'dropoff_longitude']])[:, 1]

    return df, pca

def one_hot_encode(df, encoder=None):
    if not encoder:
        encoder = OneHotEncoder(sparse=False)
    
    oneHotEncoded = encoder.fit_transform(df[["passenger_count", "year", "weekday", "hour"]])
    dfOneHotEncoded = pd.DataFrame(oneHotEncoded)
    df = pd.concat([df, dfOneHotEncoded], axis=1)
    df = df.drop(['passenger_count', 'year', 'weekday', 'hour'], axis=1)
    
    return df, encoder

def plot_predictions(y_test, y_pred):
    ''' Plot prediction and actual data
    '''
    plt.figure(figsize=(10,10))
    plt.subplot(1, 1, 1)
    plt.plot(y_test, y_pred, '.', markersize=1.5)
    plt.title('Actual fare vs Predicted fare (max $80)'), 
    plt.xlabel('Actual fare')
    plt.ylabel('Predicted fare')
    plt.xlim(0, 80)
    plt.ylim(0, 80)

    plt.show()