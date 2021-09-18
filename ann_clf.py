import pandas as pd
from IPython.display import Markdown, display
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
from sklearn import metrics
from joblib import load, dump
from utils import *
from slicefinder import *
from sklearn.neural_network import MLPClassifier
from test import *

ann_relu = load('./ann_relu.m')

column_names = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num', 'marital-status', 'occupation',
                'relationship', 'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                'income']
adult = pd.read_csv('adult_complete.csv', header=0, names=column_names, engine='python')
for col in set(adult.columns) - set(adult.describe().columns):
    adult[col] = adult[col].astype('category')

adult_data = adult.drop(columns=['income'])
adult_label = adult.income

# print(adult['income'][0:2])
# res = adult['gender'].isin(['Male'])
# print(adult['gender'].isin(['Male']))
# print(adult[~res])

# res = divide_slice(adult, flag=False)
# print(adult.iloc[0:2])
# a(data=adult, features_dict=res)

# '''
adult_cat_1hot = pd.get_dummies(adult_data.select_dtypes('category'))
adult_non_cat = adult_data.select_dtypes(exclude='category')

adult_data_1hot = pd.concat([adult_non_cat, adult_cat_1hot], axis=1, join='inner')

scaler = get_standardscaler(adult_data_1hot)
# '''
train_data, test_data = data_normalization(scaler=scaler,
                                           train_data=adult_data_1hot,
                                           test_data=adult_data_1hot)
train_label = adult_label
test_label = adult_label
# '''

feature_dict = divide_slice(adult, )
whole_feature_dict = divide_slice(adult, flag=False)

res = slice_finder(data=adult,
                   label='income',
                   features_dict=feature_dict, 
                   model=ann_relu, 
                   avg_acc=0.83,
                   scaler=scaler,
                   whole_features_dict=whole_feature_dict)
print(res)


# print(adult.iloc[1,'age'])
# adult.iloc[1]['age'] = 40
# print(adult.iloc[1]['age'])
# print(robustness_score(data=adult.iloc[0:2],
#                        label='income',
#                        ori_pred=ann_relu.predict(test_data[[0,2]]),
#                        model=ann_relu,
#                        scaler=scaler,
#                        features_dict=whole_feature_dict))
# print(res['age'][0])
# print(random.sample(divide_slice(adult, flag=False)['age'], 1))
# print(pred[0:2])
# print(test_label[0:2])
# print(model_eval(test_label, pred))
# '''
