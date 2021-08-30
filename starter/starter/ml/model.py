from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
import pickle



# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    lr = LogisticRegression(solver='lbfgs', max_iter=100)
    lr.fit(X_train, y_train)

    return lr

def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)

    return preds

def save_model(model, file_path):
    '''
    Save the trained model to a file.
    Inputs
    ------
    model : model
        Trained machine learning model.
    file_path : str
        Path to save the model.
    
    Returns:
    ________
      None
    '''
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)

def load_model(file_path):
    '''
    Load the trained model from a file.
    Inputs
    ------
    file_path : str
        Path to load the model from.
    Returns:
    ________
      model : model
    '''
    with open(file_path, 'rb') as f:
        model = pickle.load(f)

    return model
