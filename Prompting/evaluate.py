import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def evaluate_model(df, results):
    # Get true labels from the dataframe
    #y_true = df["deceptive"].tolist()
    y_true = df["label"].tolist()
    y_pred = results

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted', pos_label=None)
    conf_matrix = confusion_matrix(y_true, y_pred)

    return accuracy, f1, conf_matrix