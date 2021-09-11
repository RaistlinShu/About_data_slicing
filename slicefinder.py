import itertools
import pandas as pd
from utils import model_eval, data_normalization


def robustness_measurement(data, label, ori_pred, model, scaler, features_dict):
    wa = 0
    ac = 0
    robust_num = 1
    features_num = len(features_dict)
    features = list(features_dict.keys())

    for i in range(data.shape[0]):
        for j in range(3):
            robust_features = random.sample(features, robust_num)
            for feature in robust_features:
                if isinstance(features_dict[feature][0], tuple):
                    for k in range(len(features_dict[feature])):
                        if data.iloc[i][feature] >= min(features_dict[feature][k]) & \
                                data.iloc[i][feature] < max(features_dict[feature][k]):
                            data.iloc[i][feature] = \
                                data.iloc[i][feature] + abs(features_dict[feature][0] - features_dict[feature][1])
                else:
                    original_feature = data.iloc[i][feature]
                    while data.iloc[i][feature] == original_feature:
                        data.iloc[i][feature] = random.sample(features_dict[feature][0], 1)

            data_data = data[i: i + 1].drop(columns=[label])

            data_cat_1hot = pd.get_dummies(data_data.select_dtypes('category'))
            data_non_cat = data_data.select_dtypes(exclude='category')

            data_data_1hot = pd.concat([data_non_cat, data_cat_1hot], axis=1, join='inner')

            _, test_data = data_normalization(scaler=scaler,
                                              train_data=data_data_1hot,
                                              test_data=data_data_1hot)
            pred = model.predict(test_data)[0]
            if pred == ori_pred[i]:
                ac = ac + 1
            else:
                wa = wa + 1
    return ac / wa + ac

# import time
def score_function(data_slice, label, model, avg_acc, scaler):
    slice_data = data_slice.drop(columns=[label])
    slice_label = data_slice[label]

    slice_cat_1hot = pd.get_dummies(slice_data.select_dtypes('category'))
    slice_non_cat = slice_data.select_dtypes(exclude='category')
    adult_data_1hot = pd.concat([slice_non_cat, slice_cat_1hot], axis=1, join='inner')
    _, test_data = data_normalization(scaler, adult_data_1hot, adult_data_1hot)
    test_label = slice_label

    slice_acc = model_eval(actual=test_label, pred=model.predict(test_data))
    if slice_acc >= (0.75 * avg_acc):
        return True
    return False


def slice_finder(data, features_dict, model, avg_acc, scaler):
    slices = []
    # ['age', 'workclass', 'fnlwgt', 'education', .....]
    features = list(features_dict.keys())
    # Choose f_num features
    for f_num in range(1, len(features) + 1):
        # p is the combinations of f_nums features, eg. ('age', 'gender')
        for p in itertools.combinations(features, f_num):
            #  eg. q = ((17, 20), 'Male')
            for q in itertools.product(*[features_dict[p[k]] for k in range(f_num)]):
                target_data_idx = pd.Series([True for i in range(data.shape[0])])
                for i in range(f_num):
                    if isinstance(q[i], tuple):  # Numerical feature
                        target_data_idx = target_data_idx & \
                                          (data[p[i]] >= min(q[i])) \
                                          & (data[p[i]] < max(q[i]))
                    elif q[i].lower().find('other') == -1:  # The feature does not contain 'Other'
                        target_data_idx = target_data_idx & \
                                          (data[p[i]] == q[i])
                    else:  # contain 'Other'
                        target_data_idx = target_data_idx & \
                                          ((data[p[i]] == q[i])
                                           | ~(data[p[i]].isin(features_dict[p[i]]))
                                           )

                if score_function(data_slice=data[target_data_idx],
                                  label='income',
                                  model=model,
                                  avg_acc=avg_acc,
                                  scaler=scaler):
                    slices.append({p[num]: q[num] for num in range(f_num)})
            '''
            model_eval(target_value)
            '''
    return slices

# slice_finder()
