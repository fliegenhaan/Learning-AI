"""
Main Script - Interactive ML Pipeline for Student Dropout Prediction

Script ini merupakan interface utama untuk melatih model machine learning.
User dapat memilih model yang ingin digunakan (Logistic Regression, SVM, atau Decision Tree)
dan model akan dilatih dengan preprocessing pipeline lengkap yang sudah dioptimasi.

Workflow:
1. Load dan preprocess data (outlier removal, PCA, normalization, SMOTE)
2. User memilih model yang ingin dilatih
3. Model dilatih dan dievaluasi
4. User dapat menyimpan model untuk prediksi nantinya
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time

# Import custom modules untuk data loading dan preprocessing
from data import DataLoader, EDA
from preprocessing import DataCleaner, ModelPipeline


def load_and_prepare_data():
    """
    Load data dari Google Drive dan lakukan preprocessing lengkap

    Tahapan preprocessing:
    - Load data training dan testing
    - Exploratory Data Analysis (EDA)
    - Split data menjadi train dan validation set
    - Data cleaning (remove duplicates, handle missing values)
    - Build preprocessing pipeline (outlier removal, PCA, normalization)
    - Handle imbalanced data dengan SMOTE

    Return data yang sudah siap untuk training model
    """

    print("\n" + "=" * 50)
    print("STUDENT DROPOUT PREDICTION")
    print("=" * 50)

    # ============================================================================
    # STEP 1: LOAD DATA
    # ============================================================================
    # Data diload dari Google Drive (dataset student dropout prediction)
    print("\n[STEP 1] Loading data...")

    train_url = "https://drive.google.com/uc?id=1wzTvPSwjAK5PN0iCWEXy92_tim-5ggjs"
    test_url = "https://drive.google.com/uc?id=1ZoKNPeUAIIFIZHoKaY6_4R_fUqDue0HM"

    loader = DataLoader(train_url=train_url, test_url=test_url)
    train_df, test_df = loader.load_data()

    # ============================================================================
    # STEP 2: EXPLORATORY DATA ANALYSIS
    # ============================================================================
    # Tampilkan informasi dasar dataset (shape, missing values, distribusi target)
    print("\n[STEP 2] Exploratory data analysis...")

    eda = EDA(train_df)
    eda.show_basic_info()

    # ============================================================================
    # STEP 3: SPLIT DATA
    # ============================================================================
    # Split data menjadi features (X) dan target (y)
    # Kemudian split lagi menjadi training set (80%) dan validation set (20%)
    # Stratify digunakan untuk memastikan proporsi class seimbang di train dan val
    print("\n[STEP 3] Splitting data...")

    X, y = loader.split_features_target(train_df, target_col='Target')
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape: {X_val.shape}")

    # ============================================================================
    # STEP 4: DATA CLEANING
    # ============================================================================
    # Check missing values dan remove duplicate rows
    # Duplicate rows dapat menyebabkan overfitting dan data leakage
    print("\n[STEP 4] Data cleaning...")

    cleaner = DataCleaner()
    cleaner.check_missing_values(X_train, X_val)
    X_train, y_train = cleaner.remove_duplicates(X_train, y_train)

    # ============================================================================
    # STEP 5: PREPROCESSING PIPELINE
    # ============================================================================
    # Build pipeline lengkap yang terdiri dari:
    # 1. Outlier Removal (IQR method dengan factor 1.5)
    # 2. PCA untuk dimensionality reduction (retain 95% variance)
    # 3. Normalizer untuk standardisasi feature scale
    print("\n[STEP 5] Building preprocessing pipeline...")

    pipeline_builder = ModelPipeline()
    pipeline = pipeline_builder.build_pipeline(
        outlier_factor=1.5,      # Outlier threshold (nilai standar IQR)
        pca_components=0.95,     # Retain 95% variance
        use_normalizer=True      # PowerTransformer untuk distribusi normal
    )

    # Fit pipeline pada training data, transform both train dan validation
    # PENTING: Validation data di-transform menggunakan parameter dari training data
    # untuk menghindari data leakage
    X_train_prep, X_val_prep = pipeline_builder.fit_transform_pipeline(X_train, X_val)

    # ============================================================================
    # STEP 6: HANDLE IMBALANCED DATA
    # ============================================================================
    # Gunakan SMOTE (Synthetic Minority Over-sampling Technique) untuk handle
    # class imbalance pada training data. SMOTE membuat synthetic samples
    # untuk minority class sehingga class distribution lebih seimbang.
    # CATATAN: SMOTE hanya diterapkan pada training data, tidak pada validation!
    print("\n[STEP 6] Handling imbalanced data (SMOTE)...")

    X_train_final, y_train_final = pipeline_builder.handle_imbalanced_data(
        X_train_prep,
        y_train,
        random_state=42
    )

    # Validation data tidak di-SMOTE, tetap menggunakan distribusi asli
    X_val_final = X_val_prep
    y_val_final = y_val

    print("\nData preparation complete!")
    print(f"  Training data shape: {X_train_final.shape}")
    print(f"  Validation data shape: {X_val_final.shape}")

    return X_train_final, y_train_final, X_val_final, y_val_final, test_df, pipeline_builder


def train_logistic_regression(X_train, y_train, X_val, y_val):
    """
    Train Logistic Regression model from scratch

    Model ini menggunakan implementasi from scratch dengan:
    - Gradient descent optimization
    - L2 regularization untuk prevent overfitting
    - One-vs-Rest strategy untuk multiclass classification
    - Learning rate 0.01 dan 1000 iterations (sudah dioptimasi)
    """
    from model.logres import MultiClassLogisticRegression

    print("\n[TRAINING] Logistic Regression")

    start_time = time.time()

    # Inisialisasi model dengan hyperparameter yang sudah dioptimasi
    model = MultiClassLogisticRegression(
        learning_rate=0.01,      # Learning rate untuk gradient descent
        n_iterations=1000,       # Jumlah iterasi training
        regularization='l2',     # L2 regularization (Ridge)
        lambda_reg=0.01,         # Regularization strength
        verbose=True             # Print progress selama training
    )

    # Fit model pada training data
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Evaluate model performance pada validation set
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    # Tampilkan hasil training
    print(f"\nTraining complete ({train_time:.2f}s)")
    print(f"  Validation Accuracy: {accuracy:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_val, y_pred))

    return model, accuracy, train_time


def train_svm(X_train, y_train, X_val, y_val):
    """
    Train Support Vector Machine model from scratch

    Model ini menggunakan implementasi PEGASOS (Primal Estimated sub-GrAdient)
    yang jauh lebih cepat daripada dual formulation (O(N) vs O(N^3)).
    Support multiple kernels: linear, RBF, polynomial.
    Multiclass menggunakan One-vs-All strategy.
    """
    from model.svm import MulticlassSVM

    print("\n[TRAINING] Support Vector Machine (SVM)")

    # User memilih jenis kernel yang akan digunakan
    # - Linear: Untuk data yang linearly separable
    # - RBF: Untuk data non-linear (paling umum digunakan)
    # - Polynomial: Untuk data dengan polynomial decision boundary
    print("\nPilih kernel untuk SVM:")
    print("1. Linear")
    print("2. RBF (Radial Basis Function)")
    print("3. Polynomial")

    kernel_choice = input("\nPilihan kernel (1/2/3) [default: 2]: ").strip() or "2"

    kernel_map = {
        "1": "linear",
        "2": "rbf",
        "3": "poly"
    }
    kernel = kernel_map.get(kernel_choice, "rbf")

    print(f"\nMenggunakan kernel: {kernel}")

    start_time = time.time()
    model = MulticlassSVM(
        C=1.0,
        kernel=kernel,
        gamma='scale',
        max_iter=500,
        strategy='ova',
        verbose=True
    )
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Evaluate
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    print(f"\nTraining complete ({train_time:.2f}s)")
    print(f"  Kernel: {kernel.upper()}")
    print(f"  Validation Accuracy: {accuracy:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_val, y_pred))

    return model, accuracy, train_time


def train_decision_tree(X_train, y_train, X_val, y_val):
    """Train Decision Tree model"""
    from model.dtl import DecisionTree

    print("\n[TRAINING] Decision Tree")

    # User memilih criterion untuk splitting
    # - Gini: Mengukur impurity (lebih cepat, umumnya cukup baik)
    # - Entropy: Mengukur information gain (lebih slow tapi kadang lebih akurat)
    print("\nPilih criterion untuk Decision Tree:")
    print("1. Gini")
    print("2. Entropy")

    criterion_choice = input("\nPilihan criterion (1/2) [default: 1]: ").strip() or "1"
    criterion = "gini" if criterion_choice == "1" else "entropy"

    print(f"\nMenggunakan criterion: {criterion}")

    start_time = time.time()

    # Inisialisasi Decision Tree dengan hyperparameter untuk prevent overfitting
    model = DecisionTree(
        min_samples_split=20,    # Minimum samples untuk split node (pruning)
        max_depth=10,            # Maximum depth tree (pruning)
        criterion=criterion      # Splitting criterion yang dipilih user
    )

    # Training bisa memakan waktu karena recursive tree building
    print("Training Decision Tree (ini mungkin memakan waktu)...")
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Evaluate model performance
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    # Tampilkan hasil training
    print(f"\nTraining complete ({train_time:.2f}s)")
    print(f"  Criterion: {criterion.upper()}")
    print(f"  Validation Accuracy: {accuracy:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_val, y_pred))

    # ============================================================================
    # VISUALISASI TREE (OPSIONAL)
    # ============================================================================
    # User dapat memvisualisasikan struktur tree untuk memahami decision rules
    # Visualisasi menampilkan nodes (decision rules) dan edges (True/False branches)
    # User juga bisa membatasi kedalaman yang ditampilkan (top-N levels)
    visualize = input("\nVisualisasikan struktur tree? (y/n) [default: n]: ").strip().lower() or "n"
    if visualize == 'y':
        print("\nParameter visualisasi:")
        # max_depth_vis = top-N levels yang akan ditampilkan
        # Berguna jika tree terlalu besar dan sulit dibaca
        max_depth_vis = input("  Maksimal kedalaman tree yang ditampilkan (kosongkan untuk semua): ").strip()
        max_depth_vis = int(max_depth_vis) if max_depth_vis.isdigit() else None

        filename = input("  Nama file output [default: decision_tree.png]: ").strip() or "decision_tree.png"
        if not filename.endswith('.png'):
            filename += '.png'

        print(f"\nMembuat visualisasi tree...")
        model.visualize_tree(max_depth=max_depth_vis, filename=filename)

    return model, accuracy, train_time


def save_model(model, model_name):
    """
    Save trained model ke file .pkl

    Model disimpan di current directory agar mudah diakses.
    Format pickle digunakan agar bisa di-load kembali untuk prediksi.
    """
    save_choice = input("\nSimpan model? (y/n) [default: y]: ").strip().lower() or "y"

    if save_choice == 'y':
        try:
            filename = f"{model_name.lower().replace(' ', '_')}_model"
            # Save to current directory (not storage/)
            model.save_model(filename, format='pkl', save_dir='.')
            print(f"\nModel berhasil disimpan sebagai: {filename}.pkl")
            print(f"   Lokasi: Current directory")
        except Exception as e:
            print(f"\n✗ Gagal menyimpan model: {e}")
    else:
        print("\nModel tidak disimpan.")


def main():
    """Main function dengan interactive menu"""

    # Load and prepare data
    X_train, y_train, X_val, y_val, test_df, pipeline = load_and_prepare_data()

    # MODEL SELECTION MENU
    print("\n[STEP 7] Model Selection")

    print("\nPilih model yang ingin digunakan:")
    print("1. Logistic Regression")
    print("2. Support Vector Machine (SVM)")
    print("3. Decision Tree")
    print("4. Train All & Compare")
    print("5. Exit")

    choice = input("\nPilihan Anda (1/2/3/4/5): ").strip()

    if choice == "1":
        model, accuracy, train_time = train_logistic_regression(X_train, y_train, X_val, y_val)
        save_model(model, "logistic_regression")

        return {
            'model': model,
            'model_name': 'Logistic Regression',
            'accuracy': accuracy,
            'train_time': train_time,
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'test_df': test_df,
            'pipeline': pipeline
        }

    elif choice == "2":
        model, accuracy, train_time = train_svm(X_train, y_train, X_val, y_val)
        save_model(model, "svm")

        return {
            'model': model,
            'model_name': 'SVM',
            'accuracy': accuracy,
            'train_time': train_time,
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'test_df': test_df,
            'pipeline': pipeline
        }

    elif choice == "3":
        model, accuracy, train_time = train_decision_tree(X_train, y_train, X_val, y_val)
        save_model(model, "decision_tree")

        return {
            'model': model,
            'model_name': 'Decision Tree',
            'accuracy': accuracy,
            'train_time': train_time,
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'test_df': test_df,
            'pipeline': pipeline
        }

    elif choice == "4":
        print("\n[TRAINING ALL MODELS]")

        models_results = {}

        # Train all models
        print("\nTraining 1/3: Logistic Regression")
        logres_model, logres_acc, logres_time = train_logistic_regression(X_train, y_train, X_val, y_val)
        models_results['Logistic Regression'] = {
            'model': logres_model,
            'accuracy': logres_acc,
            'train_time': logres_time
        }

        # Train model 2: SVM dengan RBF kernel (default, umumnya perform terbaik)
        print("\nTraining 2/3: SVM (RBF kernel)")
        from model.svm import MulticlassSVM
        svm_model = MulticlassSVM(C=1.0, kernel='rbf', gamma='scale', max_iter=500, strategy='ova', verbose=True)
        start_time = time.time()
        svm_model.fit(X_train, y_train)
        svm_time = time.time() - start_time
        svm_pred = svm_model.predict(X_val)
        svm_acc = accuracy_score(y_val, svm_pred)
        print(f"\nSVM Accuracy: {svm_acc:.4f}, Time: {svm_time:.2f}s")
        models_results['SVM'] = {
            'model': svm_model,
            'accuracy': svm_acc,
            'train_time': svm_time
        }

        # Train model 3: Decision Tree dengan Gini criterion (default)
        print("\nTraining 3/3: Decision Tree")
        from model.dtl import DecisionTree
        dtl_model = DecisionTree(min_samples_split=20, max_depth=10, criterion='gini')
        start_time = time.time()
        dtl_model.fit(X_train, y_train)
        dtl_time = time.time() - start_time
        dtl_pred = dtl_model.predict(X_val)
        dtl_acc = accuracy_score(y_val, dtl_pred)
        print(f"\nDecision Tree Accuracy: {dtl_acc:.4f}, Time: {dtl_time:.2f}s")
        models_results['Decision Tree'] = {
            'model': dtl_model,
            'accuracy': dtl_acc,
            'train_time': dtl_time
        }

        # ========== COMPARISON RESULTS ==========
        # Bandingkan performance semua model yang sudah dilatih
        # Metrics: Accuracy dan Training Time
        print("\n" + "=" * 40)
        print("MODEL COMPARISON")
        print("=" * 40)
        print(f"{'Model':<25} {'Accuracy':<12} {'Time (s)':<10}")
        print("-" * 40)

        for model_name, results in models_results.items():
            print(f"{model_name:<25} {results['accuracy']:<12.4f} {results['train_time']:<10.2f}")

        # Tentukan model terbaik berdasarkan validation accuracy
        best_model_name = max(models_results, key=lambda x: models_results[x]['accuracy'])
        best_accuracy = models_results[best_model_name]['accuracy']

        print("-" * 40)
        print(f"\nBest: {best_model_name} ({best_accuracy:.4f})")

        # ========== VISUALISASI DECISION TREE (OPSIONAL) ==========
        # Jika Decision Tree termasuk dalam models yang dilatih,
        # tanya user apakah mau visualisasi struktur tree-nya
        if 'Decision Tree' in models_results:
            visualize = input("\nVisualisasikan Decision Tree? (y/n) [default: n]: ").strip().lower() or "n"
            if visualize == 'y':
                print("\nParameter visualisasi:")
                # User dapat membatasi kedalaman yang ditampilkan jika tree terlalu besar
                max_depth_vis = input("  Maksimal kedalaman tree yang ditampilkan (kosongkan untuk semua): ").strip()
                max_depth_vis = int(max_depth_vis) if max_depth_vis.isdigit() else None

                filename = input("  Nama file output [default: decision_tree_all.png]: ").strip() or "decision_tree_all.png"
                if not filename.endswith('.png'):
                    filename += '.png'

                print(f"\nMembuat visualisasi tree...")
                models_results['Decision Tree']['model'].visualize_tree(max_depth=max_depth_vis, filename=filename)

        # ========== SAVE BEST MODEL ==========
        # Simpan model terbaik untuk digunakan pada prediksi nantinya
        save_model(models_results[best_model_name]['model'], f"best_{best_model_name}")

        return {
            'models': models_results,
            'best_model_name': best_model_name,
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'test_df': test_df,
            'pipeline': pipeline
        }

    elif choice == "5":
        print("\nExiting...")
        return None

    else:
        print("\nPilihan tidak valid. Exiting...")
        return None


if __name__ == "__main__":
    print("\n")
    results = main()

    if results:
        print("\nPipeline completed successfully!")
