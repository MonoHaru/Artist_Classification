import os
import pandas as pd
from sklearn.model_selection import train_test_split

# load csv data file
def load_csv(root='./data/train.csv'):
    df = pd.read_csv(root)

    X_train, X_val, Y_train, Y_val = train_test_split(df, df['artist'].values, test_size=0.2)
    print('Number of posters for training: ', len(X_train))
    print('Number of posters for validation: ', len(X_val))

    X_train = path_preproceesing(X_train)
    X_val = path_preproceesing(X_val)
    Y_train = path_preproceesing(Y_train)
    Y_val = path_preproceesing(Y_val)

# data image path preprocessing
def path_preproceesing(data):
    data = data.sort_value(by=['id'])
    data['img_path'] = ['D:\\dacon_artist\\data\\' + path[2:] for path in data['img_path']]
    return data


