"""
Testing SVM dengan dataset student dropout

Implementasi menggunakan algoritma PEGASOS (Primal Estimated sub-GrAdient SOlver for SVM)
Reference: Shalev-Shwartz et al. (2011)

PEGASOS advantages:
- Complexity: O(N) vs O(N^3) untuk dual QP formulation
- Training speed: ~3000x lebih cepat untuk dataset berukuran 3000+ samples
- Full NumPy vectorization untuk performa optimal
- Support untuk multiple kernels: linear, RBF, polynomial
- Multiclass dengan One-vs-All (OvA) dan One-vs-One (OvO) strategy
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.svm import SVC as SklearnSVM
import matplotlib.pyplot as plt
import seaborn as sns
import time

from data import DataLoader
from preprocessing import DataCleaner, ModelPipeline
from model import SVM, MulticlassSVM


def persiapan_data():
    """memuat dan preprocessing data"""
    print("\n[PERSIAPAN DATA]")

    train_url = "https://drive.google.com/uc?id=1wzTvPSwjAK5PN0iCWEXy92_tim-5ggjs"
    test_url = "https://drive.google.com/uc?id=1ZoKNPeUAIIFIZHoKaY6_4R_fUqDue0HM"

    loader = DataLoader(train_url=train_url, test_url=test_url)
    train_df, test_df = loader.load_data()

    X, y = loader.split_features_target(train_df, target_col='Target')
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nshape data:")
    print(f"X_train: {X_train.shape}")
    print(f"X_val: {X_val.shape}")

    cleaner = DataCleaner()
    X_train, y_train = cleaner.remove_duplicates(X_train, y_train)

    pipeline_builder = ModelPipeline()
    pipeline = pipeline_builder.build_pipeline(
        outlier_factor=1.5,
        pca_components=0.95,
        use_normalizer=True
    )

    X_train_prep, X_val_prep = pipeline_builder.fit_transform_pipeline(X_train, X_val)

    X_train_final, y_train_final = pipeline_builder.handle_imbalanced_data(
        X_train_prep, y_train, random_state=42
    )

    print(f"\nshape setelah preprocessing:")
    print(f"X_train: {X_train_final.shape}")
    print(f"X_val: {X_val_prep.shape}")

    return X_train_final, y_train_final, X_val_prep, y_val, test_df, pipeline_builder


def test_svm_linear(X_train, y_train, X_val, y_val):
    """test SVM dengan kernel linear (PEGASOS)"""
    print("\n[TEST] SVM - Kernel Linear (PEGASOS)")

    start_time = time.time()

    svm = MulticlassSVM(
        C=1.0,
        kernel='linear',
        tol=1e-3,
        max_iter=200,  # Optimal untuk dataset ini (balance speed vs accuracy)
        strategy='ova',
        verbose=True
    )

    print("\nmelatih model...")
    svm.fit(X_train, y_train)

    train_time = time.time() - start_time
    print(f"\nwaktu training: {train_time:.2f} detik")

    y_train_pred = svm.predict(X_train)
    y_val_pred = svm.predict(X_val)

    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)

    print(f"\nakurasi training: {train_acc:.4f}")
    print(f"akurasi validation: {val_acc:.4f}")

    print("\nclassification report (validation):")
    print(classification_report(y_val, y_val_pred))

    print("\nconfusion matrix (validation):")
    print(confusion_matrix(y_val, y_val_pred))

    svm.save_model('svm_linear', format='pkl')
    return svm, val_acc


def test_svm_rbf(X_train, y_train, X_val, y_val):
    """test SVM dengan kernel RBF (PEGASOS)"""
    print("\n[TEST] SVM - Kernel RBF (PEGASOS)")

    start_time = time.time()

    svm = MulticlassSVM(
        C=1.0,
        kernel='rbf',
        gamma='auto',  # Auto: 1/n_features
        tol=1e-3,
        max_iter=100,  # Lebih rendah untuk kernel RBF (lebih lambat tapi tetap cukup)
        strategy='ova',
        verbose=True
    )

    print("\nmelatih model...")
    svm.fit(X_train, y_train)

    train_time = time.time() - start_time
    print(f"\nwaktu training: {train_time:.2f} detik")

    y_train_pred = svm.predict(X_train)
    y_val_pred = svm.predict(X_val)

    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)

    print(f"\nakurasi training: {train_acc:.4f}")
    print(f"akurasi validation: {val_acc:.4f}")

    print("\nclassification report (validation):")
    print(classification_report(y_val, y_val_pred))

    print("\nconfusion matrix (validation):")
    print(confusion_matrix(y_val, y_val_pred))

    svm.save_model('svm_rbf', format='pkl')
    return svm, val_acc


def test_svm_poly(X_train, y_train, X_val, y_val):
    """test SVM dengan kernel polynomial (PEGASOS)"""
    print("\n[TEST] SVM - Kernel Polynomial (PEGASOS)")

    start_time = time.time()

    svm = MulticlassSVM(
        C=1.0,
        kernel='poly',
        degree=3,
        gamma='auto',  # Auto: 1/n_features
        coef0=1.0,
        tol=1e-3,
        max_iter=100,  # Lebih rendah untuk kernel polynomial
        strategy='ova',
        verbose=True
    )

    print("\nmelatih model...")
    svm.fit(X_train, y_train)

    train_time = time.time() - start_time
    print(f"\nwaktu training: {train_time:.2f} detik")

    y_train_pred = svm.predict(X_train)
    y_val_pred = svm.predict(X_val)

    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)

    print(f"\nakurasi training: {train_acc:.4f}")
    print(f"akurasi validation: {val_acc:.4f}")

    print("\nclassification report (validation):")
    print(classification_report(y_val, y_val_pred))

    print("\nconfusion matrix (validation):")
    print(confusion_matrix(y_val, y_val_pred))

    svm.save_model('svm_poly', format='pkl')
    return svm, val_acc


def test_svm_ovo(X_train, y_train, X_val, y_val):
    """test SVM dengan strategi One-vs-One (PEGASOS)"""
    print("\n[TEST] SVM - Strategi One-vs-One (PEGASOS)")

    start_time = time.time()

    svm = MulticlassSVM(
        C=1.0,
        kernel='rbf',
        gamma='auto',  # Auto: 1/n_features
        tol=1e-3,
        max_iter=100,  # Lebih rendah untuk OvO (lebih banyak classifier)
        strategy='ovo',
        verbose=True
    )

    print("\nmelatih model...")
    svm.fit(X_train, y_train)

    train_time = time.time() - start_time
    print(f"\nwaktu training: {train_time:.2f} detik")

    y_train_pred = svm.predict(X_train)
    y_val_pred = svm.predict(X_val)

    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)

    print(f"\nakurasi training: {train_acc:.4f}")
    print(f"akurasi validation: {val_acc:.4f}")

    print("\nclassification report (validation):")
    print(classification_report(y_val, y_val_pred))

    print("\nconfusion matrix (validation):")
    print(confusion_matrix(y_val, y_val_pred))

    svm.save_model('svm_ovo', format='pkl')
    return svm, val_acc


def train_sklearn_svm(X_train, y_train, X_val, y_val, kernel='rbf', C=1.0):
    """Train SVM dengan sklearn untuk comparison"""
    print(f"\n[TRAINING] Sklearn SVM (Kernel={kernel.upper()})")

    start_time = time.time()

    # Konfigurasi SVM sesuai kernel
    if kernel == 'linear':
        model_sklearn = SklearnSVM(
            C=C,
            kernel='linear',
            max_iter=1000,
            random_state=42
        )
    elif kernel == 'rbf':
        model_sklearn = SklearnSVM(
            C=C,
            kernel='rbf',
            gamma='auto',
            max_iter=1000,
            random_state=42
        )
    elif kernel == 'poly':
        model_sklearn = SklearnSVM(
            C=C,
            kernel='poly',
            degree=3,
            gamma='auto',
            coef0=1.0,
            max_iter=1000,
            random_state=42
        )

    print("\nmelatih model sklearn...")
    model_sklearn.fit(X_train, y_train)

    train_time = time.time() - start_time
    print(f"waktu training: {train_time:.2f} detik")

    # Evaluate
    print("\nEvaluation - Sklearn:")

    y_train_pred = model_sklearn.predict(X_train)
    y_val_pred = model_sklearn.predict(X_val)

    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)

    print(f"\nakurasi training: {train_acc:.4f}")
    print(f"akurasi validation: {val_acc:.4f}")

    print("\nValidation Metrics (Sklearn):")
    print(f"Accuracy:  {accuracy_score(y_val, y_val_pred):.4f}")
    print(f"Precision: {precision_score(y_val, y_val_pred, average='weighted', zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_val, y_val_pred, average='weighted', zero_division=0):.4f}")
    print(f"F1-Score:  {f1_score(y_val, y_val_pred, average='weighted', zero_division=0):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_val, y_val_pred, zero_division=0))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_val, y_val_pred)
    print(cm)

    return model_sklearn, y_val_pred, train_time


def compare_svm_models(model_scratch, model_sklearn, X_val, y_val, kernel='rbf'):
    """Compare performance from scratch vs sklearn"""
    print(f"\n[COMPARISON] From Scratch vs Sklearn (Kernel={kernel.upper()})")

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
    visualize_svm_comparison(comparison_df, kernel)

    return comparison_df


def visualize_svm_comparison(comparison_df, kernel='rbf'):
    """Visualize SVM model comparison"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Side-by-side comparison
    comparison_df[['From Scratch', 'Sklearn']].plot(
        kind='bar', ax=axes[0], rot=0, color=['#FF6B6B', '#4ECDC4']
    )
    axes[0].set_title(f'SVM Comparison: From Scratch vs Sklearn (Kernel={kernel.upper()})',
                     fontweight='bold', fontsize=12)
    axes[0].set_ylabel('Score', fontsize=11)
    axes[0].legend(['From Scratch (PEGASOS)', 'Sklearn (LibSVM)'], fontsize=10)
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
    filename = f'svm_comparison_{kernel}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to '{filename}'")
    plt.show()


def main():
    """fungsi utama untuk testing SVM"""
    print("\n" + "=" * 50)
    print("TESTING SVM - STUDENT DROPOUT PREDICTION")
    print("=" * 50)

    X_train, y_train, X_val, y_val, test_df, pipeline = persiapan_data()

    results = {}

    model_linear, acc_linear = test_svm_linear(X_train, y_train, X_val, y_val)
    results['linear'] = acc_linear

    model_rbf, acc_rbf = test_svm_rbf(X_train, y_train, X_val, y_val)
    results['rbf'] = acc_rbf

    model_poly, acc_poly = test_svm_poly(X_train, y_train, X_val, y_val)
    results['poly'] = acc_poly

    model_ovo, acc_ovo = test_svm_ovo(X_train, y_train, X_val, y_val)
    results['ovo'] = acc_ovo

    print("\n[SUMMARY] From Scratch Results:")
    print(f"SVM Linear (OvA):      {acc_linear:.4f}")
    print(f"SVM RBF (OvA):         {acc_rbf:.4f}")
    print(f"SVM Polynomial (OvA):  {acc_poly:.4f}")
    print(f"SVM RBF (OvO):         {acc_ovo:.4f}")

    best_model = max(results, key=results.get)
    print(f"\nmodel terbaik: SVM {best_model} dengan akurasi {results[best_model]:.4f}")

    # COMPARISON WITH SKLEARN
    print("\n[COMPARISON WITH SKLEARN]")

    # Test RBF kernel (biasanya paling baik)
    print("\n--- Testing RBF Kernel ---")
    model_sklearn_rbf, y_pred_sklearn_rbf, time_sklearn_rbf = train_sklearn_svm(
        X_train, y_train, X_val, y_val, kernel='rbf', C=1.0
    )
    comparison_rbf = compare_svm_models(model_rbf, model_sklearn_rbf, X_val, y_val, kernel='rbf')

    # Test Linear kernel
    print("\n--- Testing Linear Kernel ---")
    model_sklearn_linear, y_pred_sklearn_linear, time_sklearn_linear = train_sklearn_svm(
        X_train, y_train, X_val, y_val, kernel='linear', C=1.0
    )
    comparison_linear = compare_svm_models(model_linear, model_sklearn_linear, X_val, y_val, kernel='linear')

    print("\nTesting selesai!")
    print("\nKey Comparisons:")
    print(f"  RBF Kernel:")
    print(f"    - From Scratch Accuracy: {comparison_rbf.loc['Accuracy', 'From Scratch']:.4f}")
    print(f"    - Sklearn Accuracy:      {comparison_rbf.loc['Accuracy', 'Sklearn']:.4f}")
    print(f"    - Difference:            {comparison_rbf.loc['Accuracy', 'Difference']:.4f}")
    print(f"\n  Linear Kernel:")
    print(f"    - From Scratch Accuracy: {comparison_linear.loc['Accuracy', 'From Scratch']:.4f}")
    print(f"    - Sklearn Accuracy:      {comparison_linear.loc['Accuracy', 'Sklearn']:.4f}")
    print(f"    - Difference:            {comparison_linear.loc['Accuracy', 'Difference']:.4f}")


if __name__ == "__main__":
    main()
