"""
Data Preprocessing Module
Handles feature scaling, encoding, normalization, dimensionality reduction,
and handling imbalanced datasets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, PowerTransformer
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from typing import Tuple, Optional


class ImputerManager:
    """Manages imputation strategies for numerical and categorical features"""

    def __init__(self):
        """Initialize imputer manager"""
        self.num_imputer = None
        self.cat_imputer = None

    def get_numerical_imputer(self, strategy: str = 'median') -> SimpleImputer:
        """
        Get imputer for numerical features

        Args:
            strategy: Imputation strategy ('mean', 'median', 'most_frequent', 'constant')

        Returns:
            SimpleImputer for numerical features
        """
        self.num_imputer = SimpleImputer(strategy=strategy)
        print(f"Numerical Imputer created with strategy: {strategy}")
        return self.num_imputer

    def get_categorical_imputer(self, strategy: str = 'most_frequent') -> SimpleImputer:
        """
        Get imputer for categorical features

        Args:
            strategy: Imputation strategy ('most_frequent', 'constant')

        Returns:
            SimpleImputer for categorical features
        """
        self.cat_imputer = SimpleImputer(strategy=strategy)
        print(f"Categorical Imputer created with strategy: {strategy}")
        return self.cat_imputer


class ScalerManager:
    """Manages feature scaling transformers"""

    def __init__(self):
        """Initialize scaler manager"""
        self.scaler = None

    def get_standard_scaler(self) -> StandardScaler:
        """
        Get StandardScaler (mean=0, std=1)

        Returns:
            StandardScaler instance
        """
        self.scaler = StandardScaler()
        print("StandardScaler created.")
        return self.scaler

    def get_minmax_scaler(self, feature_range: tuple = (0, 1)) -> MinMaxScaler:
        """
        Get MinMaxScaler

        Args:
            feature_range: Desired range of transformed data

        Returns:
            MinMaxScaler instance
        """
        self.scaler = MinMaxScaler(feature_range=feature_range)
        print(f"MinMaxScaler created with range: {feature_range}")
        return self.scaler


class EncoderManager:
    """Manages categorical encoding transformers"""

    def __init__(self):
        """Initialize encoder manager"""
        self.encoder = None

    def get_onehot_encoder(self, handle_unknown: str = 'ignore', sparse_output: bool = False) -> OneHotEncoder:
        """
        Get OneHotEncoder for categorical features

        Args:
            handle_unknown: How to handle unknown categories during transform
            sparse_output: Whether to return sparse matrix or dense array

        Returns:
            OneHotEncoder instance
        """
        self.encoder = OneHotEncoder(handle_unknown=handle_unknown, sparse_output=sparse_output)
        print("OneHotEncoder created.")
        return self.encoder


class NormalizerManager:
    """Manages data normalization transformers"""

    def __init__(self):
        """Initialize normalizer manager"""
        self.normalizer = None

    def get_power_transformer(self, method: str = 'yeo-johnson', standardize: bool = True) -> PowerTransformer:
        """
        Get PowerTransformer for normalization

        Args:
            method: Transformation method ('yeo-johnson' or 'box-cox')
            standardize: Whether to apply zero-mean, unit-variance normalization

        Returns:
            PowerTransformer instance
        """
        self.normalizer = PowerTransformer(method=method, standardize=standardize)
        print(f"PowerTransformer created with method: {method}")
        return self.normalizer


class DimensionalityReducer:
    """Manages dimensionality reduction transformers"""

    def __init__(self):
        """Initialize dimensionality reducer"""
        self.reducer = None

    def get_pca(self, n_components: float = 0.95, random_state: int = 42) -> PCA:
        """
        Get PCA for dimensionality reduction

        Args:
            n_components: Number of components or variance ratio to keep
            random_state: Random state for reproducibility (Kaggle requirement)

        Returns:
            PCA instance
        """
        self.reducer = PCA(n_components=n_components, random_state=random_state)
        self.reducer.set_output(transform="pandas")
        print(f"PCA created with n_components: {n_components}, random_state: {random_state}")
        return self.reducer


class ImbalanceHandler:
    """Handles imbalanced datasets using resampling techniques"""

    def __init__(self):
        """Initialize imbalance handler"""
        self.sampler = None

    def check_class_distribution(self, y_train: pd.Series, plot: bool = True) -> None:
        """
        Check and visualize class distribution

        Args:
            y_train: Target variable
            plot: Whether to plot the distribution
        """
        print("Distribusi Kelas Target (y_train):")
        print(y_train.value_counts())
        print("\nPersentase:")
        print(y_train.value_counts(normalize=True))

        if plot:
            plt.figure(figsize=(6, 4))
            sns.countplot(x=y_train)
            plt.title("Distribusi Kelas Target")
            plt.show()

    def get_smote(self, random_state: int = 42, sampling_strategy: str = 'auto') -> SMOTE:
        """
        Get SMOTE for oversampling minority classes

        Args:
            random_state: Random state for reproducibility
            sampling_strategy: Sampling strategy ('auto', 'minority', 'all', etc.)

        Returns:
            SMOTE instance
        """
        self.sampler = SMOTE(random_state=random_state, sampling_strategy=sampling_strategy)
        print("SMOTE sampler created.")
        return self.sampler

    def apply_smote(self, X_train: pd.DataFrame, y_train: pd.Series,
                    smote: Optional[SMOTE] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply SMOTE to balance the dataset

        Args:
            X_train: Training features
            y_train: Training target
            smote: SMOTE instance (if None, creates a new one)

        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        if smote is None:
            smote = self.get_smote()

        print("\nShape sebelumnya:", X_train.shape, y_train.shape)
        print("Distribusi target sebelumnya:\n", y_train.value_counts())

        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        print("\nShape sampling ulang:", X_resampled.shape, y_resampled.shape)
        print("Distribusi target sampling ulang:\n", y_resampled.value_counts())

        return X_resampled, y_resampled


class DataPreprocessor:
    """
    Main class orchestrating all preprocessing operations
    """

    def __init__(self):
        """Initialize data preprocessor with all managers"""
        self.imputer_manager = ImputerManager()
        self.scaler_manager = ScalerManager()
        self.encoder_manager = EncoderManager()
        self.normalizer_manager = NormalizerManager()
        self.dimensionality_reducer = DimensionalityReducer()
        self.imbalance_handler = ImbalanceHandler()

    def get_imputer_manager(self) -> ImputerManager:
        """Get imputer manager"""
        return self.imputer_manager

    def get_scaler_manager(self) -> ScalerManager:
        """Get scaler manager"""
        return self.scaler_manager

    def get_encoder_manager(self) -> EncoderManager:
        """Get encoder manager"""
        return self.encoder_manager

    def get_normalizer_manager(self) -> NormalizerManager:
        """Get normalizer manager"""
        return self.normalizer_manager

    def get_dimensionality_reducer(self) -> DimensionalityReducer:
        """Get dimensionality reducer"""
        return self.dimensionality_reducer

    def get_imbalance_handler(self) -> ImbalanceHandler:
        """Get imbalance handler"""
        return self.imbalance_handler
