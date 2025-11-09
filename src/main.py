import sys,os
import argparse
import kagglehub
import pandas as pd
from preprocessing.feature_engineering import feature_engineering
from models.baseline import BaselineModel
from models.main_model import MainModel
from evaluation.eval import run_evaluation
from preprocessing.data_preprocessing import preprocess_data
from preprocessing.data_preprocessing import normalize_data
from evaluation.eval import run_evaluation, compare_models



"""
Main execution script for training, tuning, and evaluating models.

Usage:
    python src/main.py [options]
    
Options:
    -h, --hyperparameter_tune : Perform hyperparameter tuning before training the model.
    -ms, --model_save_path : Path to save the trained model. Default is 'trained_model.joblib'.
    -mp, --model_load_path : Path to load the model from. Default is 'trained_model.joblib'.
"""

# TODO : Add more command line arguments for different functionalities
def get_args():
    parser = argparse.ArgumentParser(
        description="Train, tune, and evaluate the asteroid classification model."
    )

    # Optional flag: hyperparameter tuning
    parser.add_argument(
        '-ht','--hyperparameter_tune',  # no short '-h', to avoid clash with help
        action='store_true',
        help='Perform hyperparameter tuning before training the model.'
    )

    # Optional: paths for saving/loading model
    parser.add_argument(
        '--model_save_path',
        type=str,
        default='trained_model.joblib',
        help='Path to save the trained model.'
    )
    parser.add_argument(
        '--model_load_path',
        type=str,
        default='trained_model.joblib',
        help='Path to load the model from.'
    )
    parser.add_argument(
        '--feature_set',
        type=str,
        default='all',
        help=(
            "Which feature set to use for training. Options: "
            "'all', 'physical', 'orbital_core', 'orbital_derived', 'motion_timing', 'uncertainties', 'model_fit', "
            "or a combination separated by commas (e.g., 'physical,orbital_core')."
        )
    )

    return parser.parse_args()

def download_dataset():
    """Download dataset from Kaggle if not already in data/raw."""
    raw_path = "data/raw/dataset.csv"
    if not os.path.exists(raw_path):
        print("Downloading dataset from Kaggle...")
        dataset_dir = kagglehub.dataset_download("sakhawat18/asteroid-dataset")
        source_path = os.path.join(dataset_dir, "dataset.csv")
        os.makedirs("data/raw", exist_ok=True)
        os.system(f"cp '{source_path}' '{raw_path}'")
        print(f"Dataset saved to {raw_path}")
    else:
        print("Raw dataset already exists.")
    return raw_path

def check_preprocessed():
    """Check if preprocessed dataset exists; if not, run preprocessing."""
    preprocessed_path = "data/preprocessed/asteroid_clean.csv"
    raw_path = "data/raw/dataset.csv"

    if not os.path.exists(preprocessed_path):
        print("Preprocessed dataset not found. Running preprocessing...")

        # Ensure folder exists
        os.makedirs(os.path.dirname(preprocessed_path), exist_ok=True)

        # Call your preprocessing function (which should save the file)
        preprocess_data(raw_path, preprocessed_path)

        print(f"Preprocessed data saved to {preprocessed_path}")
    else:
        print("Preprocessed dataset already exists.")

    return preprocessed_path

if __name__ == "__main__":
    
    args = get_args()

    # Ensure dataset exists
    download_dataset()
    preprocessed_csv = check_preprocessed()


    # After downloading and preprocessing
    df = pd.read_csv(preprocessed_csv)


    # Initialize feature engineer
    engineer = feature_engineering(target_col='pha')

    # Call select_features passing args.feature_set
    X_train, y_train, X_test, y_test = engineer.select_features(df, feature_set=args.feature_set)

    #print("NaNs in y_train:", y_train.isna().sum())
    #print("NaNs in y_test:", y_test.isna().sum())
    

    print("Data ready!")
    print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}, Features: {X_train.shape[1]}")

    # Print all feature names for inspection
    print("Selected feature names:")
    print(list(X_train.columns))
    print("Selected target col:")
    print(y_train.name)

    X_train_norm, X_test_norm,  =  normalize_data(X_train, X_test)


    #models
    baseline = BaselineModel()
    baseline.fit(y_train)
    mm = MainModel()

    # Hyperparameter tuning if asked for in args
    if args.hyperparameter_tune:
        mm.hyperparameter_tuning(X_train_norm, y_train)

    # Get model (load from file if exists, else create new)
    model, is_loaded = mm.get_model(load_path=args.model_load_path)

    # Trains the model (if not loaded from file) and saves it
    if not is_loaded:
        model = mm.train(model, X_train_norm, y_train, save_path=args.model_save_path)

    # Run models (run on X_test, returns y_pred)
    y_pred_main_model = mm.predict(model, X_test_norm)

    y_baseline_pred = baseline.predict(X_test_norm)

    # Evaluate each model individually
    results_baseline = run_evaluation(y_test, y_baseline_pred, model_name="Baseline Model", cmap="Blues")
    results_main = run_evaluation(y_test, y_pred_main_model, model_name="Main Model", cmap="Greens")

    # Compare them visually and numerically
    compare_models(results_baseline, results_main)
  























   