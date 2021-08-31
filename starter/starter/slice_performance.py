# this is to compare the slice performance of the model

import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import load_model, inference, compute_model_metrics


def compute_slice_performance(fixed_feature, fixed_value):
    '''
    Compute the model performance for a given slice

    Inputs
    ------
    fixed_feature: str
        Name of the category
    fixed_value: str
        Value of the category
    Returns
    -------
    precision, recall, fbeta
    '''

    data = pd.read_csv('../data/clean_census.csv')

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    # Optional enhancement,
    # use K-fold cross validation instead of a train-test split.
    _, test = train_test_split(data, test_size=0.30)

    test = test[test[fixed_feature] == fixed_value]

    model = load_model('../model/model.pkl')
    encoder = load_model('../model/encoder.pkl')
    lb = load_model('../model/lb.pkl')

    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    y_preds = inference(model, X_test)
    return compute_model_metrics(y_test, y_preds)


precision, recall, fbeta = compute_slice_performance('sex', 'Male')
precision2, recall2, fbeta2 = compute_slice_performance('sex', 'Female')

with open('slice_output.txt', 'w') as f:
    print('Category: sex, Value: Male', file=f)
    print('- Precision: ' + str(precision), file=f)
    print('- Recall: ' + str(recall), file=f)
    print('- Fbeta: ' + str(fbeta), file=f)

    print('Category: sex, Value: Female', file=f)
    print('- Precision: ' + str(precision2), file=f)
    print('- Recall: ' + str(recall2), file=f)
    print('- Fbeta: ' + str(fbeta2), file=f)
