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

    test['X_test'] = X_test
    print(test[0])
    # test_feature_fixed = test[test[fixed_feature] == fixed_value]

    # y_preds = inference(model, test_feature_fixed['X_test'])
    # return compute_model_metrics(y_test, y_preds)


precision, recall, fbeta = compute_slice_performance('workclass', ' Private')

print('precision', precision)
print('recall', recall)
print('fbeta', fbeta)
