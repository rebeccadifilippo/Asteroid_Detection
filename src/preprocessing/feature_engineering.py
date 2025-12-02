import pandas as pd
from sklearn.model_selection import train_test_split

class feature_engineering:
    def __init__(self, target_col='pha'):
        self.target_col = target_col

        # Define feature groups
        self.feature_groups = {
            'physical': ['H', 'diameter', 'albedo', 'diameter_sigma'],
            'orbital_core': ['e', 'a', 'q', 'i', 'om', 'w', 'ma'],
            'orbital_derived': ['ad', 'moid', 'moid_ld'],
            'motion_timing': ['n', 'tp', 'per', 'per_y'],
            'uncertainties': [
                'sigma_e','sigma_a','sigma_q','sigma_i','sigma_om','sigma_w','sigma_ma',
                'sigma_ad','sigma_n','sigma_tp','sigma_per'
            ],
            'model_fit': ['rms']
        }

    def select_features(self, df, feature_set='all', test_size=0.2, random_state=42):
        # Convert pha and neo columns to 0/1 safely
        for col in ['pha', 'neo']:
            if col in df.columns:
                df[col] = df[col].astype(float).astype(int)
        # Select features
        if feature_set.lower() == 'all':
            X = df.drop(columns=[self.target_col])
        else:
            feature_batches = [fs.strip() for fs in feature_set.split(',')]
            selected_cols = []
            for batch in feature_batches:
                cols = [c for c in self.feature_groups.get(batch, []) if c in df.columns]
                selected_cols.extend(cols)
            X = df[selected_cols]

        # Target
        y = df[self.target_col]

        
        
        # Simple train/test split for testing (no stratify)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y # Maintain class distribution
        )

        return X_train, y_train, X_test, y_test

