import math
import os
import glob
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer, MinMaxScaler
from sklearn.pipeline import Pipeline


def get_dataset(path='cicids2017-original/TrafficLabelling', test_size=0.5):
    # path = 'cicids2017/'
    # path = 'cicids2017-original/TrafficLabelling'
    all_files = glob.glob(os.path.join(path, "*.csv"))
    print(all_files)
    df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

    print(df.size)

    df[' Label'] = df[' Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

    Target = ' Label'
    X = df.drop(columns=[Target])
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], [np.finfo(np.float64).max, np.finfo(np.float64).min])
    pipeline = Pipeline([('normalizer', Normalizer()), ('scaler', MinMaxScaler())])
    X = pipeline.fit_transform(X)

    y = df[Target]

    return train_test_split(X, y, test_size=test_size)

def get_dataset_autoencoders(path='cicids2017-original/TrafficLabelling', test_size=0.5):
    # path = 'cicids2017/'
    # path = 'cicids2017-original/TrafficLabelling'
    all_files = glob.glob(os.path.join(path, "*.csv"))
    print(all_files)
    df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

    print(df.size)
    target = ' Label'

    df[target] = df[target].apply(lambda x: 0 if x == 'BENIGN' else 1)

    benign = df[df[target] == 0].sample(frac=1).reset_index(drop=True)
    attack = df[df[target] == 1]

    X_train = benign.iloc[:math.ceil(len(df)*(1-test_size))]
    X_test = pd.concat([benign.iloc[math.ceil(len(df)*(1-test_size)):], attack]).sample(frac=1)

    X_train = X_train.drop(columns=[target])
    X_train, X_validate = train_test_split(X_train, test_size=0.2, random_state=42)

    X_test, Y_test = X_test.drop(columns=[target]), X_test[target]

    X_train, X_validate, X_test = normalize_data(X_train), normalize_data(X_validate), normalize_data(X_test)

    return X_train, X_validate, X_test, Y_test


def normalize_data(df):
    X = df.fillna(0)
    X = X.replace([np.inf, -np.inf], [np.finfo(np.float64).max, np.finfo(np.float64).min])

    pipeline = Pipeline([('normalizer', Normalizer()), ('scaler', MinMaxScaler())])
    X = pipeline.fit_transform(X)

    return X