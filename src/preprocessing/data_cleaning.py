"""
Data Cleaning Module
Handles outliers, duplicates, and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Tuple


class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Custom transformer to handle outliers using IQR method
    Outliers are replaced with the median value
    """

    def __init__(self, factor: float = 1.5):
        """
        Initialize OutlierHandler

        Args:
            factor: IQR multiplier for outlier detection (default: 1.5)
        """
        self.factor = factor
        self.lower_bound_ = {}
        self.upper_bound_ = {}
        self.median_ = {}
        self.numeric_cols_ = None

    def fit(self, X, y=None):
        """
        Fit the outlier handler by calculating bounds and medians

        Args:
            X: Input features (DataFrame)
            y: Target (not used)

        Returns:
            self
        """
        self.numeric_cols_ = X.select_dtypes(include=[np.number]).columns

        for col in self.numeric_cols_:
            data = X[col]
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1

            self.lower_bound_[col] = Q1 - self.factor * IQR
            self.upper_bound_[col] = Q3 + self.factor * IQR
            self.median_[col] = data.median()

        return self

    def transform(self, X):
        """
        Transform data by replacing outliers with median

        Args:
            X: Input features (DataFrame)

        Returns:
            Transformed DataFrame
        """
        X_out = X.copy()

        for col in self.numeric_cols_:
            if col in X_out.columns:
                lower = self.lower_bound_[col]
                upper = self.upper_bound_[col]
                median = self.median_[col]

                outliers = (X_out[col] < lower) | (X_out[col] > upper)
                if outliers.sum() > 0:
                    X_out.loc[outliers, col] = median

        return X_out


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for feature engineering
    Creates new features from existing ones
    """

    def __init__(self):
        """Initialize FeatureEngineer"""
        pass

    def fit(self, X, y=None):
        """
        Fit method (no fitting required for this transformer)

        Args:
            X: Input features
            y: Target (not used)

        Returns:
            self
        """
        return self

    def transform(self, X):
        """
        Transform data by creating new features

        Args:
            X: Input features (DataFrame)

        Returns:
            Transformed DataFrame with new features
        """
        X_eng = X.copy()

        # 1. Binning Age (Age at enrollment)
        # Grouping age into categories: Teen, Adult, Senior
        if 'Age at enrollment' in X_eng.columns:
            bins = [0, 20, 30, 100]
            labels = ['Teen', 'Adult', 'Senior']
            X_eng['Age group'] = pd.cut(X_eng['Age at enrollment'], bins=bins, labels=labels)
            X_eng['Age group'] = X_eng['Age group'].astype('object')

        return X_eng


class DataCleaner:
    """
    Main class for data cleaning operations
    Handles missing values, duplicates, outliers, and feature engineering
    """

    def __init__(self):
        """Initialize DataCleaner"""
        self.has_missing = False
        self.duplicate_count = 0

    def check_missing_values(self, X_train: pd.DataFrame, X_val: pd.DataFrame) -> bool:
        """
        Check for missing values in train and validation sets

        Args:
            X_train: Training features
            X_val: Validation features

        Returns:
            True if missing values found, False otherwise
        """
        train_missing = X_train.isnull().sum().sum()
        val_missing = X_val.isnull().sum().sum()

        if train_missing == 0 and val_missing == 0:
            self.has_missing = False
            print("Tidak ada missing value pada train_set dan val_set\n")
        else:
            self.has_missing = True
            print(f"Missing values found - Train: {train_missing}, Val: {val_missing}\n")

        print("Train missing values:\n", X_train.isnull().sum())
        print("\nVal missing values:\n", X_val.isnull().sum())

        return self.has_missing

    def remove_duplicates(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Remove duplicate rows from training set

        Args:
            X_train: Training features
            y_train: Training target

        Returns:
            Tuple of (X_train_cleaned, y_train_cleaned)
        """
        duplicates = X_train.duplicated().sum()
        self.duplicate_count = duplicates
        print(f"Number of duplicate rows in X_train: {duplicates}")

        if duplicates > 0:
            duplicate_indices = X_train[X_train.duplicated()].index
            X_train = X_train.drop(index=duplicate_indices)
            y_train = y_train.drop(index=duplicate_indices)

            print(f"Removed {duplicates} duplicate rows.")
            print(f"New shape of X_train: {X_train.shape}")
            print(f"New shape of y_train: {y_train.shape}")
        else:
            print("No duplicates found.")

        return X_train, y_train

    def get_outlier_handler(self, factor: float = 1.5) -> OutlierHandler:
        """
        Get an OutlierHandler instance

        Args:
            factor: IQR multiplier for outlier detection

        Returns:
            OutlierHandler instance
        """
        return OutlierHandler(factor=factor)

    def get_feature_engineer(self) -> FeatureEngineer:
        """
        Get a FeatureEngineer instance

        Returns:
            FeatureEngineer instance
        """
        return FeatureEngineer()
