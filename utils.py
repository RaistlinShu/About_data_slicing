import pandas as pd
from sklearn.preprocessing import StandardScaler


def model_eval(actual, pred):
    confusion = pd.crosstab(actual, pred, rownames=['Actual'], colnames=['Predicted'])
    TP = confusion.loc['>50K', '>50K']
    TN = confusion.loc['<=50K', '<=50K']
    FP = confusion.loc['<=50K', '>50K']
    FN = confusion.loc['>50K', '<=50K']

    accuracy = (TP + TN) / (TP + FN + FP + TN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f_measure = (2 * recall * precision) / (recall + precision)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    error_rate = 1 - accuracy

    out = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f_measure': f_measure,
           'sensitivity': sensitivity, 'specificity': specificity, 'error_rate': error_rate}

    return out


def get_standardscaler(original_train_data):
    scaler = StandardScaler()

    # Fitting only on training data
    scaler.fit(original_train_data)
    return scaler


def data_normalization(scaler, train_data, test_data):
    # Transform the train data
    train_data = scaler.transform(train_data)

    # Applying same transformation to test data
    test_data = scaler.transform(test_data)
    return train_data, test_data


def create_bins(lower_bound, upper_bound, quantity):
    """
    create bins for numeric
    """
    width = (upper_bound - lower_bound) / quantity
    bins = list()
    for i in range(quantity - 1):
        bins.append((int(lower_bound + i * width), int(lower_bound + (i + 1) * width)))
    bins.append((int(lower_bound + (quantity - 1) * width), upper_bound + 1))
    return bins


def notfindother(values):
    for x in values:
        if x.lower().find('other') != -1:
            return False
    return True


def divide_slice(data, min_slice_size=100, flag=True):
    res = {}
    data_size = data.shape[0]
    if flag:
        size_threshold = data_size / min_slice_size
    else:
        size_threshold = 1
    for column in data:
        if str(data[column].dtype) == 'category':
            values = list()
            value_counts = data[column].value_counts()
            index = value_counts.index
            # print(index)
            flag = 0
            for i in range(len(index)):
                if value_counts[index[i]] >= size_threshold:
                    values.append(index[i])
                else:
                    flag = 1
            if flag == 1 and notfindother(values=values):
                values.append('Other')
            res[str(column)] = values
        else:
            res[str(column)] = create_bins(lower_bound=data[column].min(),
                                           upper_bound=data[column].max(),
                                           quantity=5)
    return res
