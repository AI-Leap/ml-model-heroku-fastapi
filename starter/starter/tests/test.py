import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
from starter.starter.ml.model \
    import load_model, inference, compute_model_metrics
from starter.starter.ml.data import process_data


def test_load_model():
    model = load_model('./starter/model/model.pkl')
    assert isinstance(model, LogisticRegression)


def test_inference():
    data = pd.read_csv('./starter/data/clean_census.csv')

    _, test = train_test_split(data, test_size=0.20)

    cat_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]

    model = load_model('./starter/model/model.pkl')
    encoder = load_model('./starter/model/encoder.pkl')
    lb = load_model('./starter/model/lb.pkl')

    X_test, _, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    y_preds = inference(model, X_test)
    assert isinstance(y_preds, (np.ndarray, np.generic))


def test_compute_model_metrics():
    data = pd.read_csv('./starter/data/clean_census.csv')

    # Optional enhancement,
    # use K-fold cross validation instead of a train-test split.
    _, test = train_test_split(data, test_size=0.20)

    cat_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]

    model = load_model('./starter/model/model.pkl')
    encoder = load_model('./starter/model/encoder.pkl')
    lb = load_model('./starter/model/lb.pkl')

    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    y_preds = inference(model, X_test)

    precsion, recall, fbeta = compute_model_metrics(y_test, y_preds)

    assert isinstance(precsion, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)

    assert precsion > 0.0 and precsion <= 1.0
    assert recall > 0.0 and recall <= 1.0
    assert fbeta > 0.0 and fbeta <= 1.0
