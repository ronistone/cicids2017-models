import os
import glob
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def get_dataset(path='cicids2017-original/TrafficLabelling', test_size=0.5):
    # path = 'cicids2017/'
    # path = 'cicids2017-original/TrafficLabelling'
    all_files = glob.glob(os.path.join(path, "*.csv"))
    print(all_files)
    df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

    print(df.size)

    df[' Label'] = df[' Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    # df[' Label'] = LabelEncoder().fit_transform(df[' Label'])

    Target = ' Label'
    # webattack_features = ['Average Packet Size', 'Flow Bytes/s', 'Max Packet Length', 'Packet Length Mean',
    #                       'Fwd Packet Length Mean', 'Subflow Fwd Bytes', 'Fwd IAT Min', 'Avg Fwd Segment Size',
    #                       'Total Length of Fwd Packets', 'Fwd IAT Std', 'Fwd Packet Length Max', 'Flow IAT Mean',
    #                       'Fwd Header Length', 'Flow Duration', 'Flow Packets/s', 'Fwd IAT Mean',
    #                       'Fwd IAT Total', 'Fwd Packets/s', 'Flow IAT Std', 'Fwd IAT Max']
    X = df.drop(columns=[Target, 'Flow Bytes/s', ' Flow Packets/s'])
    # X = df[webattack_features]
    x_columns = X.columns
    # X = (X - X.mean()) / X.std()
    # X = np.log(X)
    # print(X.isnull().sum()[:60])
    # print(X.isna().sum()[:60])
    # print('max:\n', X.max()[:60])
    # print('\n\nmin:\n', X.min()[:60])

    y = df[Target]

    # imputer = SimpleImputer(strategy='mean')
    # X = imputer.fit_transform(X)

    # transformer = IncrementalPCA(n_components=70, batch_size=10000)
    # X = transformer.fit_transform(X)
    # print(X.shape)

    return train_test_split(X, y, test_size=test_size)
