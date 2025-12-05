"""
Test Final - Logistic Regression
Mengintegrasikan preprocessing pipeline lengkap dengan Logistic Regression training
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Import custom modules
from data import DataLoader, EDA
from preprocessing import DataCleaner, ModelPipeline
from model.logres import LogisticRegression, MultiClassLogisticRegression


def load_and_preprocess_data():
    """
    Load data dan apply preprocessing pipeline lengkap
    Sama seperti main.py workflow
    """
    print("\n[DATA LOADING & PREPROCESSING]")

    # Load data
    train_url = "https://drive.google.com/uc?id=1wzTvPSwjAK5PN0iCWEXy92_tim-5ggjs"
    test_url = "https://drive.google.com/uc?id=1ZoKNPeUAIIFIZHoKaY6_4R_fUqDue0HM"

    loader = DataLoader(train_url=train_url, test_url=test_url)
    train_df, test_df = loader.load_data()

    print(f"\nData loaded: {train_df.shape}")
    print(f"Target distribution:\n{train_df['Target'].value_counts()}")

    # Split features and target
    X, y = loader.split_features_target(train_df, target_col='Target')
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")

    # Data cleaning
    print("\nData cleaning...")

    cleaner = DataCleaner()
    cleaner.check_missing_values(X_train, X_val)
    X_train, y_train = cleaner.remove_duplicates(X_train, y_train)

    # Build preprocessing pipeline
    print("\nBuilding preprocessing pipeline...")

    pipeline_builder = ModelPipeline()
    pipeline = pipeline_builder.build_pipeline(
        outlier_factor=1.5,
        pca_components=0.95,
        use_normalizer=True  # PowerTransformer for normalization
    )

    # Fit and transform
    X_train_prep, X_val_prep = pipeline_builder.fit_transform_pipeline(X_train, X_val)

    # Handle imbalanced data with SMOTE
    print("\nHandling imbalanced data (SMOTE)...")

    X_train_final, y_train_final = pipeline_builder.handle_imbalanced_data(
        X_train_prep, y_train, random_state=42
    )

    print(f"\nPreprocessing complete!")
    print(f"  Training shape: {X_train_final.shape}")
    print(f"  Validation shape: {X_val_prep.shape}")

    return X_train_final, y_train_final, X_val_prep, y_val


def train_from_scratch(X_train, y_train, X_val, y_val):
    """Train Logistic Regression from scratch dengan data yang sudah preprocessed"""
    print("\n[TRAINING] Logistic Regression From Scratch")

    n_classes = len(np.unique(y_train))

    if n_classes == 2:
        print("\nBinary Classification Mode")
        model = LogisticRegression(
            learning_rate=0.1,  # Bisa lebih tinggi karena data sudah standardized
            n_iterations=2000,
            regularization='l2',
            lambda_reg=0.1,
            verbose=True
        )
    else:
        print(f"\nMulticlass Classification Mode ({n_classes} classes - One-vs-Rest)")
        model = MultiClassLogisticRegression(
            learning_rate=0.1,  # Bisa lebih tinggi karena data sudah standardized
            n_iterations=2000,
            regularization='l2',
            lambda_reg=0.1,
            verbose=True
        )

    model.fit(X_train, y_train)

    # Evaluate
    print("\nEvaluation - From Scratch:")

    train_acc = model.score(X_train, y_train)
    val_acc = model.score(X_val, y_val)

    print(f"\nTraining Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")

    # Detailed validation metrics
    y_pred_val = model.predict(X_val)

    print("\nValidation Metrics (From Scratch):")
    print(f"Accuracy:  {accuracy_score(y_val, y_pred_val):.4f}")
    print(f"Precision: {precision_score(y_val, y_pred_val, average='weighted', zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_val, y_pred_val, average='weighted', zero_division=0):.4f}")
    print(f"F1-Score:  {f1_score(y_val, y_pred_val, average='weighted', zero_division=0):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_val, y_pred_val, zero_division=0))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_val, y_pred_val)
    print(cm)

    return model, y_pred_val


def train_sklearn(X_train, y_train, X_val, y_val):
    """Train Logistic Regression dengan sklearn untuk comparison"""
    print("\n[TRAINING] Sklearn Logistic Regression")

    model_sklearn = SklearnLogisticRegression(
        max_iter=2000,
        penalty='l2',
        C=10,  # C = 1/lambda_reg
        solver='lbfgs',
        multi_class='ovr',
        random_state=42
    )

    model_sklearn.fit(X_train, y_train)

    # Evaluate
    print("\nEvaluation - Sklearn:")

    train_acc = model_sklearn.score(X_train, y_train)
    val_acc = model_sklearn.score(X_val, y_val)

    print(f"\nTraining Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")

    # Detailed validation metrics
    y_pred_val = model_sklearn.predict(X_val)

    print("\nValidation Metrics (Sklearn):")
    print(f"Accuracy:  {accuracy_score(y_val, y_pred_val):.4f}")
    print(f"Precision: {precision_score(y_val, y_pred_val, average='weighted', zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_val, y_pred_val, average='weighted', zero_division=0):.4f}")
    print(f"F1-Score:  {f1_score(y_val, y_pred_val, average='weighted', zero_division=0):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_val, y_pred_val, zero_division=0))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_val, y_pred_val)
    print(cm)

    return model_sklearn, y_pred_val


def compare_models(model_scratch, model_sklearn, X_val, y_val):
    """Compare performance from scratch vs sklearn"""
    print("\n[COMPARISON] From Scratch vs Sklearn")

    # Predictions
    y_pred_scratch = model_scratch.predict(X_val)
    y_pred_sklearn = model_sklearn.predict(X_val)

    # Compute metrics
    metrics = {
        'Accuracy': [
            accuracy_score(y_val, y_pred_scratch),
            accuracy_score(y_val, y_pred_sklearn)
        ],
        'Precision': [
            precision_score(y_val, y_pred_scratch, average='weighted', zero_division=0),
            precision_score(y_val, y_pred_sklearn, average='weighted', zero_division=0)
        ],
        'Recall': [
            recall_score(y_val, y_pred_scratch, average='weighted', zero_division=0),
            recall_score(y_val, y_pred_sklearn, average='weighted', zero_division=0)
        ],
        'F1-Score': [
            f1_score(y_val, y_pred_scratch, average='weighted', zero_division=0),
            f1_score(y_val, y_pred_sklearn, average='weighted', zero_division=0)
        ]
    }

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(metrics, index=['From Scratch', 'Sklearn']).T
    comparison_df['Difference'] = comparison_df['From Scratch'] - comparison_df['Sklearn']
    comparison_df['Diff %'] = (comparison_df['Difference'] / comparison_df['Sklearn'] * 100).round(2)

    print("\nMetrics Comparison:")
    print(comparison_df)

    # Performance summary
    print("\nPerformance Summary:")

    avg_diff = abs(comparison_df['Difference'].mean())
    print(f"\nAverage absolute difference: {avg_diff:.4f}")

    if avg_diff < 0.05:
        print("EXCELLENT: From scratch performance is very close to sklearn!")
    elif avg_diff < 0.10:
        print("GOOD: From scratch performance is comparable to sklearn")
    else:
        print("⚠ NEEDS IMPROVEMENT: Performance gap is significant")

    # Visualize comparison
    visualize_comparison(comparison_df)

    return comparison_df


def visualize_comparison(comparison_df):
    """Visualize model comparison"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Side-by-side comparison
    comparison_df[['From Scratch', 'Sklearn']].plot(
        kind='bar', ax=axes[0], rot=0, color=['#2E86AB', '#A23B72']
    )
    axes[0].set_title('Model Comparison: From Scratch vs Sklearn',
                     fontweight='bold', fontsize=12)
    axes[0].set_ylabel('Score', fontsize=11)
    axes[0].legend(['From Scratch', 'Sklearn'], fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_ylim([0, 1])

    # Plot 2: Difference
    colors = ['red' if x < 0 else 'green' for x in comparison_df['Difference']]
    comparison_df['Difference'].plot(
        kind='bar', ax=axes[1], color=colors, rot=0
    )
    axes[1].set_title('Difference (From Scratch - Sklearn)',
                     fontweight='bold', fontsize=12)
    axes[1].set_ylabel('Difference', fontsize=11)
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    axes[1].grid(True, alpha=0.3, axis='y')

    # Plot 3: Percentage difference
    colors_pct = ['red' if x < -5 else 'orange' if x < 0 else 'green'
                  for x in comparison_df['Diff %']]
    comparison_df['Diff %'].plot(
        kind='bar', ax=axes[2], color=colors_pct, rot=0
    )
    axes[2].set_title('Percentage Difference (%)',
                     fontweight='bold', fontsize=12)
    axes[2].set_ylabel('Difference (%)', fontsize=11)
    axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    axes[2].axhline(y=-5, color='orange', linestyle='--', linewidth=0.5, alpha=0.5)
    axes[2].axhline(y=5, color='orange', linestyle='--', linewidth=0.5, alpha=0.5)
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('logistic_regression_comparison_final.png', dpi=300, bbox_inches='tight')
    print("\nComparison plot saved to 'logistic_regression_comparison_final.png'")
    plt.show()


def save_model(model, filename='logistic_regression_final.pkl'):
    """Save trained model"""
    print("\n[SAVING MODEL]")

    try:
        if isinstance(model, LogisticRegression):
            model.save_model(filename, format='pickle')
            model.save_model(filename.replace('.pkl', '.json'), format='json')
        elif isinstance(model, MultiClassLogisticRegression):
            model.save_model(filename)

        print(f"Model saved successfully!")
    except Exception as e:
        print(f"Error saving model: {e}")


def main():
    """Main function - Complete workflow"""
    print("\n" + "=" * 50)
    print("LOGISTIC REGRESSION - FINAL TEST")
    print("=" * 50)

    # 1. Load and preprocess data (with full pipeline)
    X_train, y_train, X_val, y_val = load_and_preprocess_data()

    # 2. Train from scratch
    model_scratch, y_pred_scratch = train_from_scratch(X_train, y_train, X_val, y_val)

    # 3. Train with sklearn
    model_sklearn, y_pred_sklearn = train_sklearn(X_train, y_train, X_val, y_val)

    # 4. Compare models
    comparison_df = compare_models(model_scratch, model_sklearn, X_val, y_val)

    # 5. Save model
    save_model(model_scratch)

    print("\nTest completed successfully!")
    print("\nKey Results:")
    print(f"  - From Scratch Accuracy: {comparison_df.loc['Accuracy', 'From Scratch']:.4f}")
    print(f"  - Sklearn Accuracy:      {comparison_df.loc['Accuracy', 'Sklearn']:.4f}")
    print(f"  - Difference:            {comparison_df.loc['Accuracy', 'Difference']:.4f}")
    print(f"  - Diff %:                {comparison_df.loc['Accuracy', 'Diff %']:.2f}%")


if __name__ == "__main__":
    main()
