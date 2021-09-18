import itertools
import random
import pandas as pd
from utils import model_eval, data_normalization, divide_slice


def robustness_score(data, label, ori_pred, model, scaler, features_dict):
    data = data.copy().reset_index(drop=True)
    wa = 0
    ac = 0
    robust_num = 1
    features_num = len(features_dict)
    features = list(features_dict.keys())

    for i in range(data.shape[0]):
        # if i % 1000 == 0:
        #     print('+++++++++++++++++++++++++++++++++++++++++++++++++++++')
        for j in range(2):
            robust_features = random.sample(features, robust_num)
            # print(data.iloc[i])
            for feature in robust_features:
                # print(feature)
                feature = str(feature)
                if isinstance(features_dict[feature][0], tuple):
                    for k in features_dict[feature]:
                        if data.loc[i, feature] >= min(k) & \
                                data.loc[i, feature] < max(k):
                            data.loc[i, feature] = \
                                data.loc[i, feature] + abs(k[0] - k[1])
                            break
                else:
                    original_feature = data.loc[i, feature]
                    while data.loc[i, feature] == original_feature \
                            or data.loc[i, feature].lower().find('other') != -1:
                        data.loc[i, feature] = random.sample(features_dict[feature], 1)
                # print(data.iloc[i])

            data_data = data.iloc[[i]].drop(columns=[label])

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
    return ac / (wa + ac)


def score_function(data_slice, label, model, avg_acc, scaler, whole_features_dict):
    # Separate the data and label
    slice_data = data_slice.drop(columns=[label])
    slice_label = data_slice[label]

    slice_cat_1hot = pd.get_dummies(slice_data.select_dtypes('category'))
    slice_non_cat = slice_data.select_dtypes(exclude='category')
    adult_data_1hot = pd.concat([slice_non_cat, slice_cat_1hot], axis=1, join='inner')
    _, test_data = data_normalization(scaler, adult_data_1hot, adult_data_1hot)
    test_label = slice_label

    ori_pred = model.predict(test_data)
    acc_score = model_eval(actual=test_label, pred=model.predict(test_data))['accuracy']
    robust_score = robustness_score(data_slice, label='income',
                                    ori_pred=ori_pred,
                                    model=model,
                                    scaler=scaler
                                    , features_dict=whole_features_dict)
    fairness_score = min(acc_score / avg_acc, avg_acc / acc_score) if acc_score != 0 and avg_acc != 0 else 0
    score = acc_score + fairness_score + robust_score
    print('score:', score, 'acc:', acc_score, 'fairness:', fairness_score, 'robust:', robust_score)
    if score < (avg_acc * 0.75 + 0.9 + 0.75):
        print("True")
        return True
    return False


def slice_finder(data, label, features_dict, model, avg_acc, scaler, whole_features_dict):
    slices = []
    # ['age', 'workclass', 'fnlwgt', 'education', .....]
    features = list(features_dict.keys())
    # Choose f_num features
    for f_num in range(1, len(features) + 1):
    # for f_num in range(1, 2):
        # p is the combinations of f_nums features, eg. ('age', 'gender')
        for p in itertools.combinations(features, f_num):
            print(p)
            #  eg. q = ((17, 20), 'Male')
            for q in itertools.product(*[features_dict[p[k]] for k in range(f_num)]):
                print(q)
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

                candidate_data = data[target_data_idx]
                print(candidate_data.shape)
                if candidate_data.shape[0] >= (data.shape[0]) / 1000 and \
                        score_function(data_slice=candidate_data,
                                       label=label,
                                       model=model,
                                       avg_acc=avg_acc,
                                       scaler=scaler,
                                       whole_features_dict=whole_features_dict):
                    slices.append({p[num]: q[num] for num in range(f_num)})
            '''
            model_eval(target_value)
            '''

# slice_finder()
