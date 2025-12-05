"""
Test Final - Decision Tree Learning (DTL)
Mengintegrasikan preprocessing pipeline lengkap dengan DTL training
Membandingkan performa Scratch vs Scikit-Learn (CART) + VISUALISASI LENGKAP
"""

import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.tree import DecisionTreeClassifier as SklearnDTL
import matplotlib.pyplot as plt
import seaborn as sns

# Import custom modules
from data import DataLoader
from preprocessing import DataCleaner, ModelPipeline
from model.dtl import DecisionTree

def load_and_preprocess_data():
    """Load data dan apply preprocessing pipeline"""
    print("\n[DATA LOADING & PREPROCESSING]")

    # Load data
    train_url = "https://drive.google.com/uc?id=1wzTvPSwjAK5PN0iCWEXy92_tim-5ggjs"
    test_url = "https://drive.google.com/uc?id=1ZoKNPeUAIIFIZHoKaY6_4R_fUqDue0HM"

    loader = DataLoader(train_url=train_url, test_url=test_url)
    train_df, test_df = loader.load_data()

    # Split features and target
    X, y = loader.split_features_target(train_df, target_col='Target')
    
    # Split Train/Val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Cleaning
    cleaner = DataCleaner()
    X_train, y_train = cleaner.remove_duplicates(X_train, y_train)

    # Pipeline Construction
    pipeline_builder = ModelPipeline()
    pipeline = pipeline_builder.build_pipeline(
        outlier_factor=1.5,
        pca_components=0.95,
        use_normalizer=False # Tree tidak butuh normalisasi, tapi ok untuk konsistensi pipeline
    )

    # Fit Transform
    X_train_prep, X_val_prep = pipeline_builder.fit_transform_pipeline(X_train, X_val)

    # Handle Imbalance (SMOTE)
    X_train_final, y_train_final = pipeline_builder.handle_imbalanced_data(
        X_train_prep, y_train, random_state=42
    )

    print(f"Final training data shape: {X_train_final.shape}")
    return X_train_final, y_train_final, X_val_prep, y_val

def train_from_scratch(X_train, y_train, X_val, y_val):
    """Train DTL From Scratch"""
    print("\n[TRAINING] Decision Tree From Scratch (CART)")

    start_time = time.time()
    
    # Parameter disesuaikan agar robust
    model = DecisionTree(
        min_samples_split=20, 
        max_depth=10, 
        criterion='gini'
    )

    print("Melatih model (ini mungkin memakan waktu karena rekursif python)...")
    model.fit(X_train, y_train)
    
    duration = time.time() - start_time
    print(f"Training selesai dalam {duration:.2f} detik")

    # Evaluate
    print("\nEvaluation - From Scratch:")

    y_pred_val = model.predict(X_val)
    
    acc = accuracy_score(y_val, y_pred_val)
    print(f"Validation Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred_val, zero_division=0))

    return model, y_pred_val

def train_sklearn(X_train, y_train, X_val, y_val):
    """Train Scikit-Learn Decision Tree untuk perbandingan"""
    print("\n[TRAINING] Sklearn Decision Tree")

    # Sesuai spesifikasi: CART gunakan criterion='gini'
    model_sklearn = SklearnDTL(
        criterion='gini',
        min_samples_split=20,
        max_depth=10,
        random_state=42
    )

    model_sklearn.fit(X_train, y_train)

    y_pred_val = model_sklearn.predict(X_val)
    acc = accuracy_score(y_val, y_pred_val)
    print(f"Validation Accuracy: {acc:.4f}")

    return model_sklearn, y_pred_val

def visualize_comparison(comparison_df):
    """Membuat grafik perbandingan performa (Sama seperti LogReg/SVM)"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Side-by-side comparison
    # Warna: Hijau (Scratch) vs Biru (Sklearn)
    comparison_df[['From Scratch', 'Sklearn']].plot(
        kind='bar', ax=axes[0], rot=0, color=['#2d6a4f', '#4361ee']
    )
    axes[0].set_title('DTL Comparison: From Scratch vs Sklearn',
                     fontweight='bold', fontsize=12)
    axes[0].set_ylabel('Score', fontsize=11)
    axes[0].legend(['From Scratch', 'Sklearn'], fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_ylim([0, 1.05]) # Biar ada ruang di atas bar

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
    filename = 'dtl_comparison_final.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to '{filename}'")
    plt.show() # Tampilkan jika di notebook/IDE support

def compare_models(y_val, y_pred_scratch, y_pred_sklearn):
    """Bandingkan performa secara lengkap dan visualisasikan"""
    print("\n[COMPARISON] Scratch vs Sklearn")

    # Hitung 4 Metrik Utama
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
    
    # Buat DataFrame
    comparison_df = pd.DataFrame(metrics, index=['From Scratch', 'Sklearn']).T
    comparison_df['Difference'] = comparison_df['From Scratch'] - comparison_df['Sklearn']
    comparison_df['Diff %'] = (comparison_df['Difference'] / comparison_df['Sklearn'] * 100).round(2)
    
    print("\nMetrics Comparison Table:")
    print(comparison_df)
    
    # Summary Singkat
    avg_diff = abs(comparison_df['Difference'].mean())
    print(f"\nAverage Absolute Difference: {avg_diff:.4f}")
    
    if avg_diff < 0.05:
        print("EXCELLENT: From scratch performance is very close to sklearn!")
    elif avg_diff < 0.10:
        print("GOOD: From scratch performance is comparable to sklearn")
    else:
        print("⚠ NOTE: Performance gap exists (Check overfitting/hyperparameters)")

    # Panggil fungsi visualisasi
    visualize_comparison(comparison_df)

    return comparison_df

def save_model_dtl(model):
    try:
        # Gunakan format pkl saja agar aman dari error JSON
        model.save_model("dtl_final_model", format='pkl')
        print("\nModel berhasil disimpan (dtl_final_model.pkl)!")
    except Exception as e:
        print(f"\n⚠ Gagal menyimpan model: {e}")

def main():
    # 1. Load Data
    X_train, y_train, X_val, y_val = load_and_preprocess_data()

    # 2. Train Scratch
    model_scratch, y_pred_scratch = train_from_scratch(X_train, y_train, X_val, y_val)

    # 3. Train Sklearn
    model_sklearn, y_pred_sklearn = train_sklearn(X_train, y_train, X_val, y_val)

    # 4. Compare & Visualize
    compare_models(y_val, y_pred_scratch, y_pred_sklearn)

    # 5. Save
    save_model_dtl(model_scratch)

if __name__ == "__main__":
    main()