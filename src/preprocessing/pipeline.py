"""
Pipeline Module
Builds and manages the complete preprocessing pipeline
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from typing import Tuple

from .data_cleaning import OutlierHandler, FeatureEngineer
from .data_preprocessing import (
    ImputerManager,
    ScalerManager,
    EncoderManager,
    NormalizerManager,
    DimensionalityReducer,
    ImbalanceHandler
)


class ModelPipeline:
    """
    Main pipeline class that orchestrates all preprocessing steps
    """

    def __init__(self):
        """Initialize the model pipeline"""
        self.pipeline = None
        self.imputer_manager = ImputerManager()
        self.scaler_manager = ScalerManager()
        self.encoder_manager = EncoderManager()
        self.normalizer_manager = NormalizerManager()
        self.dimensionality_reducer = DimensionalityReducer()
        self.imbalance_handler = ImbalanceHandler()

        # Components
        self.missing_data_handler = None
        self.scale_encode_transformer = None

    def build_missing_data_handler(self) -> ColumnTransformer:
        """
        Build transformer for handling missing data

        Returns:
            ColumnTransformer for imputation
        """
        num_imputer = self.imputer_manager.get_numerical_imputer(strategy='median')
        cat_imputer = self.imputer_manager.get_categorical_imputer(strategy='most_frequent')

        self.missing_data_handler = ColumnTransformer(
            transformers=[
                ('num', num_imputer, selector(dtype_include=np.number)),
                ('cat', cat_imputer, selector(dtype_exclude=np.number))
            ],
            verbose_feature_names_out=False
        )
        self.missing_data_handler.set_output(transform="pandas")

        print("Missing data handler created.")
        return self.missing_data_handler

    def build_scale_encode_transformer(self, use_normalizer: bool = True) -> ColumnTransformer:
        """
        Build transformer for scaling and encoding

        Args:
            use_normalizer: If True, use PowerTransformer for normalization,
                          else use StandardScaler

        Returns:
            ColumnTransformer for scaling and encoding
        """
        if use_normalizer:
            scaler = self.normalizer_manager.get_power_transformer()
        else:
            scaler = self.scaler_manager.get_standard_scaler()

        encoder = self.encoder_manager.get_onehot_encoder()

        self.scale_encode_transformer = ColumnTransformer(
            transformers=[
                ('scaler', scaler, selector(dtype_include=np.number)),
                ('encoder', encoder, selector(dtype_exclude=np.number))
            ],
            verbose_feature_names_out=False
        )
        self.scale_encode_transformer.set_output(transform="pandas")

        print("Scale/Encode transformer created.")
        return self.scale_encode_transformer

    def build_pipeline(self, outlier_factor: float = 1.5,
                      pca_components: float = 0.95,
                      use_normalizer: bool = True) -> Pipeline:
        """
        Build the complete preprocessing pipeline

        Args:
            outlier_factor: IQR multiplier for outlier detection
            pca_components: Number of PCA components or variance ratio
            use_normalizer: Whether to use PowerTransformer (normalization)

        Returns:
            Complete sklearn Pipeline
        """
        # Build components
        self.build_missing_data_handler()
        self.build_scale_encode_transformer(use_normalizer=use_normalizer)

        # Get other components
        feature_engineer = FeatureEngineer()
        outlier_handler = OutlierHandler(factor=outlier_factor)
        pca = self.dimensionality_reducer.get_pca(n_components=pca_components, random_state=42)

        # Build pipeline
        self.pipeline = Pipeline(steps=[
            ('feature_engineering', feature_engineer),
            ('imputer', self.missing_data_handler),
            ('outlier_handler', outlier_handler),
            ('preprocessing', self.scale_encode_transformer),
            ('pca', pca)
        ])

        print("\n" + "=" * 50)
        print("PIPELINE BUILT")
        print("=" * 50)
        print("Steps:")
        print("1. Feature Engineering")
        print("2. Missing Data Imputation")
        print("3. Outlier Handling")
        print("4. Scaling/Encoding/Normalization")
        print("5. Dimensionality Reduction (PCA)")
        print("=" * 50)

        return self.pipeline

    def fit_transform_pipeline(self, X_train: pd.DataFrame, X_val: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fit and transform data using the pipeline

        Args:
            X_train: Training features
            X_val: Validation features

        Returns:
            Tuple of (X_train_transformed, X_val_transformed)
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not built. Call build_pipeline() first.")

        print("\nFitting and transforming training data...")
        X_train_prep = self.pipeline.fit_transform(X_train)

        print("Transforming validation data...")
        X_val_prep = self.pipeline.transform(X_val)

        print(f"\nX_train_prep shape: {X_train_prep.shape}")
        print(f"X_val_prep shape: {X_val_prep.shape}")

        return X_train_prep, X_val_prep

    def handle_imbalanced_data(self, X_train: pd.DataFrame, y_train: pd.Series,
                               random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply SMOTE to handle imbalanced dataset

        Args:
            X_train: Training features (preprocessed)
            y_train: Training target
            random_state: Random state for reproducibility

        Returns:
            Tuple of (X_train_resampled, y_train_resampled)
        """
        print("\n" + "=" * 50)
        print("HANDLING IMBALANCED DATA")
        print("=" * 50)

        # Check distribution before
        self.imbalance_handler.check_class_distribution(y_train, plot=True)

        # Apply SMOTE
        smote = self.imbalance_handler.get_smote(random_state=random_state)
        X_resampled, y_resampled = self.imbalance_handler.apply_smote(X_train, y_train, smote)

        return X_resampled, y_resampled

    def get_pipeline(self) -> Pipeline:
        """Get the built pipeline"""
        return self.pipeline

    def transform_test_data(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """
        Transform test data using fitted pipeline

        Args:
            X_test: Test features

        Returns:
            Transformed test features
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not built or fitted.")

        print("Transforming test data...")
        X_test_prep = self.pipeline.transform(X_test)
        print(f"X_test_prep shape: {X_test_prep.shape}")

        return X_test_prep
