"""Preprocessing utilities for credit card fraud datasets."""

from sklearn.model_selection import train_test_split

from config.config import RANDOM_STATE, TEST_SIZE


def split_features_target(dataframe, target_column):
    """Split a dataframe into feature matrix X and target vector y."""
    if target_column not in dataframe.columns:
        raise ValueError(f"Target column '{target_column}' was not found.")

    X = dataframe.drop(columns=[target_column])
    y = dataframe[target_column]
    return X, y


def create_train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Create a stratified train/test split before fitting preprocessing steps."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
