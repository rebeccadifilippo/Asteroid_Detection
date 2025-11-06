import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os

def run_baseline():
    print("=== Running Baseline Model ===")

    # Load dataset
    file_path = os.path.join(os.path.dirname(__file__), "../data/raw/dataset.csv")
    target_col = "pha"

    # Load dataset
    df = pd.read_csv(file_path, low_memory=False)

    # Map target to 0/1, drop unknowns
    df[target_col] = df[target_col].map({"Y": 1, "N": 0})
    df = df.dropna(subset=[target_col])
    df[target_col] = df[target_col].astype(int)

    # Drop non-numeric columns (id, name, etc.)
    non_numeric_cols = ['id','spkid','full_name','pdes','name','prefix','orbit_id','epoch_cal','equinox','tp_cal','class']
    X = df.drop(columns=non_numeric_cols + [target_col])
    X = X.apply(pd.to_numeric, errors='coerce')  # force numeric
    X = X.fillna(0)  # fill missing numeric values

    y = df[target_col]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Baseline classifier
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Metrics
    print("Baseline Classification Report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    run_baseline()