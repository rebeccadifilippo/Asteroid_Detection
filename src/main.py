import sys
import argparse
from models.baseline import run_baseline
from models.main_model import MainModel
from evaluation.eval import run_evaluation

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
    parser.add_argument(
        '-h', '--hyperparameter_tune',
        action='store_true',
        help='Perform hyperparameter tuning before training the model.'
    )
    parser.add_argument(
        '-ms', '--model_save_path',
        type=str,
        default='trained_model.joblib',
        help='Path to save the trained model.'
    )
    parser.add_argument(
        '-mp', '--model_load_path',
        type=str,
        default='trained_model.joblib',
        help='Path to load the model from.'
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # Preprocess (returns X_train, y_train, X_test, y_test)
    # TODO : implement data loading and preprocessing
    X_train = None
    y_train = None
    X_test = None
    y_test = None

    # Train models and save outputs (takes in X_train, y_train)
    mm = MainModel()

    # Hyperparameter tuning if asked for in args
    if args.hyperparameter_tune:
        mm.hyperparameter_tuning(X_train, y_train)

    # Get model (load from file if exists, else create new)
    model, is_loaded = mm.get_model(load_path=args.model_load_path)

    # Trains the model (if not loaded from file) and saves it
    if not is_loaded:
        model = mm.train(model, X_train, y_train, save_path=args.model_save_path)

    # Run models (run on X_test, returns y_pred)
    y_pred = mm.predict(model, X_test)

    # Evaluate results (takes in y_test, y_pred)
    run_evaluation(y_test, y_pred)
