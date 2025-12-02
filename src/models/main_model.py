import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV
import os
import joblib
import json
from scipy.stats import uniform, randint


def save_best_params(params, file_path='model_params.json'):
    """
    Save the best hyperparameters dict to a JSON file.
    The path defaults to 'model_params.json' if unspecified.
    """

    with open(file_path, 'w') as f:
        json.dump(params, f, indent=4)

    print(f"Parameters saved to {file_path}")


def load_best_params(file_path='model_params.json'):
    """
    Load the best hyperparameters dict from a JSON file.
    The path defaults to 'model_params.json' if unspecified.
    """

    if not os.path.exists(file_path):
        print(f"No parameter file found at {file_path}")
        return None

    with open(file_path, 'r') as f:
        params = json.load(f)

    print(f"Parameters loaded from {file_path}")

    return params


class MainModel:
    def generate_new_model(self, use_best_params=False, **kwargs):
        """
        Initialize and return a HistGradientBoostingClassifier with specified hyperparameters.
        The parameters can be overridden by passing them as keyword arguments.

        If use_best_params is True, the function will attempt to load the best parameters as
        saved from previous hyperparameter tuning.

        Example:
            model = generate_new_model(learning_rate=0.05, max_iter=200, ...)

        Parameters that can be modified include:
        - loss
        - learning_rate
        - max_iter
        - max_depth
        - random_state

        Relevant Loss options:
        - 'log_loss': Logistic loss for classification.
        - 'auto': Automatically selects the loss function based on the data.
        - 'binary_crossentropy': Binary cross-entropy loss for binary classification.
        """

        # **kwargs will create a dictionary of parameters to override defaults, based on provided args
        params = kwargs

        if use_best_params:
            best_params = load_best_params()

            if best_params:
                params.update(best_params)
            else:
                print("Using default parameters as no best parameters were found.")

        # Uses respective default parameters if not provided
        return HistGradientBoostingClassifier(
            loss=params.get('loss', 'log_loss'),
            learning_rate=params.get('learning_rate', 0.1),
            max_iter=params.get('max_iter', 100),
            max_depth=params.get('max_depth', 3),
            random_state=params.get('random_state', 42),
            class_weight=params.get('class_weight', 'balanced'), # Handle class imbalance
        )


    def hyperparameter_tuning(self, X_train, y_train):
        """
        Perform hyperparameter tuning using RandomizedSearchCV to find the best parameters
        for the HistGradientBoostingClassifier.
        Saves and returns the best parameters found.
        Randomized search is superior to grid search in all aspects, including speed and performance.

        Tuned Hyperparameters:
        - learning_rate: Controls the contribution of each tree to the final model.
        - max_depth: Maximum depth of the individual trees.
        - max_iter: Number of boosting iterations (trees).
        """

        param_dist = {
            'learning_rate': uniform(0.01, 0.19),  # Range: 0.01 to 0.2
            'max_depth': randint(3, 10),           # Range: 3 to 9
            'max_iter': randint(50, 251),          # Range: 50 to 250
        }

        model = self.generate_new_model()

        f2_scorer = make_scorer(fbeta_score, beta=2, pos_label=1)

        random_search = RandomizedSearchCV(
            model,
            param_distributions=param_dist, # Can choose any value from above ranges
            n_iter=100,                      # Tries 10 different combinations (Could increase this later if needed, but it will take longer)
            scoring=f2_scorer,              # Optimize for F2 score (Weights recall more heavily)
            cv=3,                           # 3-fold cross-validation
            # random_state=42,              # Uncomment for reproducibility (I disabled this for now to allow improvement across runs)
            n_jobs=-1                       # Use all available cores to parallelize fits
        )

        random_search.fit(X_train, y_train)

        best_params = random_search.best_params_
        print("Best Parameters from Hyperparameter Tuning:", best_params)

        save_best_params(best_params)

        return best_params


    def train(self, model, X_train, y_train, save_path='trained_model.joblib'):
        """
        Train the provided model using the training data.
        If save_path is provided, save the trained model to that path using joblib.
        Returns the trained model.
        """

        model.fit(X_train, y_train)

        if save_path:
            joblib.dump(model, save_path)
            print(f"Model saved to {save_path}")

        return model


    def get_model(self, load_path='trained_model.joblib'):
        """
        Load a model from the specified path if provided, otherwise create a new model.
        Returns the model and a boolean indicating whether it was loaded from a file.
        """

        if load_path and os.path.exists(load_path):
            model = joblib.load(load_path)
            print(f"Model loaded from {load_path}")
            return model, True
        else:
            model = self.generate_new_model(use_best_params=True)

            if not load_path:
                print("New model created using best known hyperparameters.")
            else:
                print(f"Model file not found at {load_path}. New model created using best known hyperparameters.")

            return model, False


    def predict(self, model, X_test):
        """
        Use the provided model to make predictions on the test data.
        Returns the predicted labels.
        """

        y_pred = model.predict(X_test)
        return y_pred


# TODO: break this function into smaller parts across separate files.
# def run_main_model():
#     raw_file = 'data/raw/dataset.csv'
#     preprocessed_file = 'data/preprocessed/asteroid_clean.csv'
#
#     # Load preprocessed dataset if exists
#     if os.path.exists(preprocessed_file):
#         print(f"Loaded preprocessed dataset: {preprocessed_file}")
#         df = pd.read_csv(preprocessed_file, low_memory=False)
#     else:
#         print(f"Preprocessed file not found. Loading raw dataset: {raw_file}")
#         df = pd.read_csv(raw_file, low_memory=False)
#
#         # Drop non-numeric columns
#         non_numeric_cols = [
#             'id', 'spkid', 'full_name', 'pdes', 'name', 'prefix',
#             'orbit_id', 'epoch_cal', 'equinox', 'tp_cal', 'class'
#         ]
#         df.drop(columns=[col for col in non_numeric_cols if col in df.columns], inplace=True)
#
#         # Convert categorical columns to numeric
#         if 'neo' in df.columns:
#             df['neo'] = df['neo'].map({'N': 0, 'Y': 1})
#         if 'pha' in df.columns:
#             df['pha'] = df['pha'].map({'N': 0, 'Y': 1})
#
#         # Force everything else to numeric, strings become NaN
#         df = df.apply(pd.to_numeric, errors='coerce')
#
#         # Fill missing values
#         df = df.fillna(df.mean())
#
#         # Save preprocessed dataset for future runs
#         df.to_csv(preprocessed_file, index=False)
#         print(f"Saved preprocessed dataset: {preprocessed_file}")
#
#     # Features and target
#     if 'pha' not in df.columns:
#         raise ValueError("Target column 'pha' not found in dataset.")
#     X = df.drop('pha', axis=1)
#     y = df['pha'].fillna(0).astype(int)  # <<< Force discrete 0/1
#
#     # Split dataset
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.01, random_state=42, stratify=y
#     )
#
#     # Train Gradient Boosting model
#     model = HistGradientBoostingClassifier(
#         max_iter=100,
#         learning_rate=0.1,
#         max_depth=3,
#         random_state=42
#     )
#     print("Training Gradient Boosting model...")
#     model.fit(X_train, y_train)
#
#     # Evaluate
#     y_pred = model.predict(X_test)
#     print("\nGradient Boosting Classification Report:\n")
#     print(classification_report(y_test, y_pred, zero_division=0))
#     print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
