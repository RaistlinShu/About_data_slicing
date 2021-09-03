import pandas as pd
from IPython.display import Markdown, display
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
from sklearn import metrics
from joblib import load, dump
from utils import *
from sklearn.neural_network import MLPClassifier

ann_relu = load('./ann_relu.m')


def printmd(string):
    display(Markdown(string))


column_names = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num', 'marital-status', 'occupation',
                'relationship', 'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                'income']
adult = pd.read_csv('adult_complete.csv', header=0, names=column_names, engine='python')
for col in set(adult.columns) - set(adult.describe().columns):
    adult[col] = adult[col].astype('category')

adult_data = adult.drop(columns=['income'])
adult_label = adult.income


# res = divide_slice(adult)
# print(res)
#
# adult_cat_1hot = pd.get_dummies(adult_data.select_dtypes('category'))
# adult_non_cat = adult_data.select_dtypes(exclude='category')
#
# adult_data_1hot = pd.concat([adult_non_cat, adult_cat_1hot], axis=1, join='inner')
#
# train_data, test_data = data_normalization(adult_data_1hot, adult_data_1hot)
# train_label = adult_label
# test_label = adult_label
#
# pred = ann_relu.predict(test_data)
# print(model_eval(test_label, pred))
