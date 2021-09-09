import itertools
import pandas as pd
from utils import model_eval, data_normalization


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
    slices = set()
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
                    slices.add({p[num]: q[num] for num in range(f_num)})
            '''
            model_eval(target_value)
            '''
    return slices

# slice_finder()
