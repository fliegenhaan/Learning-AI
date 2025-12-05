"""
Data Loader Module
Handles loading datasets from various sources
"""

import pandas as pd
from typing import Tuple


class DataLoader:
    """Class for loading and managing datasets"""

    def __init__(self, train_url: str = None, test_url: str = None):
        """
        Initialize DataLoader with URLs

        Args:
            train_url: URL or path to training data
            test_url: URL or path to test data
        """
        self.train_url = train_url
        self.test_url = test_url
        self.train_df = None
        self.test_df = None

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load training and test datasets

        Returns:
            Tuple of (train_df, test_df)
        """
        if self.train_url:
            self.train_df = pd.read_csv(self.train_url)
            print(f"Training data loaded: {self.train_df.shape}")

        if self.test_url:
            self.test_df = pd.read_csv(self.test_url)
            print(f"Test data loaded: {self.test_df.shape}")

        return self.train_df, self.test_df

    def split_features_target(self, df: pd.DataFrame, target_col: str = 'Target') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split dataframe into features and target

        Args:
            df: Input dataframe
            target_col: Name of target column

        Returns:
            Tuple of (X, y)
        """
        X = df.drop(columns=[target_col])
        y = df[target_col]
        return X, y

    def get_train_data(self) -> pd.DataFrame:
        """Get training dataframe"""
        return self.train_df

    def get_test_data(self) -> pd.DataFrame:
        """Get test dataframe"""
        return self.test_df
