"""
Exploratory Data Analysis Module
Provides visualization and statistical analysis of datasets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional


class EDA:
    """Class for performing Exploratory Data Analysis"""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize EDA with a dataframe

        Args:
            df: Input dataframe for analysis
        """
        self.df = df

    def show_basic_info(self) -> None:
        """Display basic information about the dataset"""
        print("=" * 50)
        print("DATASET INFO")
        print("=" * 50)
        print(self.df.info())

        print("\n" + "=" * 50)
        print("DATASET DESCRIPTION")
        print("=" * 50)
        print(self.df.describe())

        print("\n" + "=" * 50)
        print("MISSING VALUES")
        print("=" * 50)
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("No missing values found.")
        else:
            print(missing[missing > 0])

    def plot_target_distribution(self, target_col: str = 'Target') -> None:
        """
        Plot distribution of target variable

        Args:
            target_col: Name of target column
        """
        if target_col not in self.df.columns:
            print(f"Column '{target_col}' not found in dataframe.")
            return

        plt.figure(figsize=(8, 6))
        sns.countplot(data=self.df, x=target_col)
        plt.title('Distribusi Variabel Target')
        plt.show()

    def plot_numerical_vs_target(self, numerical_cols: List[str], target_col: str = 'Target') -> None:
        """
        Plot boxplots of numerical features vs target

        Args:
            numerical_cols: List of numerical column names
            target_col: Name of target column
        """
        n_cols = len(numerical_cols)
        fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 6))

        if n_cols == 1:
            axes = [axes]

        for i, col in enumerate(numerical_cols):
            if col in self.df.columns:
                sns.boxplot(data=self.df, x=target_col, y=col, ax=axes[i])
                axes[i].set_title(f'{col} vs Target')

        plt.tight_layout()
        plt.show()

    def plot_categorical_vs_target(self, categorical_cols: List[str], target_col: str = 'Target') -> None:
        """
        Plot countplots of categorical features vs target

        Args:
            categorical_cols: List of categorical column names
            target_col: Name of target column
        """
        for col in categorical_cols:
            if col in self.df.columns:
                plt.figure(figsize=(10, 6))
                sns.countplot(data=self.df, x=col, hue=target_col)
                plt.title(f'{col} vs Target')
                plt.xticks(rotation=45)
                plt.show()

    def plot_correlation_matrix(self, figsize: tuple = (24, 20)) -> None:
        """
        Plot correlation heatmap for numerical features

        Args:
            figsize: Figure size for the plot
        """
        numeric_df = self.df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            print("No numerical columns found.")
            return

        plt.figure(figsize=figsize)
        sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm')
        plt.title('Matriks Korelasi Kolom Numerik')
        plt.show()

    def run_full_eda(self, target_col: str = 'Target',
                     numerical_cols: Optional[List[str]] = None,
                     categorical_cols: Optional[List[str]] = None) -> None:
        """
        Run complete EDA workflow

        Args:
            target_col: Name of target column
            numerical_cols: List of numerical columns to analyze
            categorical_cols: List of categorical columns to analyze
        """
        self.show_basic_info()

        if target_col in self.df.columns:
            self.plot_target_distribution(target_col)

        if numerical_cols:
            self.plot_numerical_vs_target(numerical_cols, target_col)

        if categorical_cols:
            self.plot_categorical_vs_target(categorical_cols, target_col)

        self.plot_correlation_matrix()
