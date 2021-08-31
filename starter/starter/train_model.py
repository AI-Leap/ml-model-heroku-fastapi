# Script to train machine learning model.
from sklearn.model_selection import train_test_split
import pandas as pd

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import train_model, save_model, inference, compute_model_metrics

# Add code to load in the data.
data = pd.read_csv('../data/clean_census.csv')

# Optional enhancement,
# use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

save_model(encoder, '../model/encoder.pkl')
save_model(lb, '../model/lb.pkl')

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)

save_model(model, '../model/model.pkl')

y_preds = inference(model, X_test)

precision, recall, fbeta = compute_model_metrics(y_test, y_preds)

print('precision', precision)
print('recall', recall)
print('fbeta', fbeta)
