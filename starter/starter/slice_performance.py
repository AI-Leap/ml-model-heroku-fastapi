import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import load_model, inference, compute_model_metrics


def compute_slice_performance(fixed_feature, fixed_value):
    data = pd.read_csv('../data/census.csv')
    data.columns = [label.strip() for label in data.columns]

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
    train, test = train_test_split(data, test_size=0.20)

    model = load_model('../model/model.pkl')

    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=cat_features, label="salary", training=True
    )
    print('ts', test.shape)
    print('xs', X_test.shape)

    x_test_filtered = []
    y_test_filtered = []
    j = 0
    for i in test.index:
        if test.at[i, fixed_feature] == fixed_value:
            x_test_filtered.append(X_test[j])
            y_test_filtered.append(y_test[j])
        j += 1

    y_preds = inference(model, x_test_filtered)
    return compute_model_metrics(y_test_filtered, y_preds)


precision, recall, fbeta = compute_slice_performance('workclass', ' Private')
precision2, recall2, fbeta2 = compute_slice_performance(
    'workclass',
    ' Never-worked'
)


with open('slice_output.txt', 'w') as f:
    print('Category: workclass, Value: Private', file=f)
    print('- Precision: ' + str(precision), file=f)
    print('- Recall: ' + str(recall), file=f)
    print('- Fbeta: ' + str(fbeta), file=f)

    print('Category: workclass, Value: Never-worked', file=f)
    print('- Precision: ' + str(precision2), file=f)
    print('- Recall: ' + str(recall2), file=f)
    print('- Fbeta: ' + str(fbeta2), file=f)
