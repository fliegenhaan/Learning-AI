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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import time
from datetime import datetime

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

    # return pipeline yang sudah di-fit, bukan pipeline_builder
    return X_train_final, y_train_final, X_val_final, y_val_final, test_df, pipeline


def train_logistic_regression(X_train, y_train, X_val, y_val):
    """
    train logistic regression model from scratch

    model ini menggunakan implementasi from scratch dengan:
    - gradient descent optimization
    - l2 regularization untuk prevent overfitting
    - one-vs-rest strategy untuk multiclass classification
    - learning rate 0.01 dan 1000 iterations (sudah dioptimasi)
    """
    from model.logres import MultiClassLogisticRegression

    print("\n[TRAINING] Logistic Regression")

    start_time = time.time()

    # inisialisasi model dengan hyperparameter
    model = MultiClassLogisticRegression(
        learning_rate=0.01,      # learning rate untuk gradient descent
        n_iterations=1000,       # jumlah iterasi training
        regularization='l2',     # l2 regularization (ridge)
        lambda_reg=0.01,         # regularization strength
        verbose=True             # print progress selama training
    )

    # fit model pada training data
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    # evaluate model performance pada validation set
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    # tampilkan hasil training
    print(f"\nTraining complete ({train_time:.2f}s)")
    print(f"  Validation Accuracy: {accuracy:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_val, y_pred))

    return model, accuracy, train_time


def train_svm(X_train, y_train, X_val, y_val):
    """
    train support vector machine model from scratch

    model ini menggunakan implementasi pegasos (primal estimated sub-gradient)
    yang jauh lebih cepat daripada dual formulation (o(n) vs o(n^3)).
    support multiple kernels: linear, rbf, polynomial.
    multiclass menggunakan one-vs-all strategy.
    """
    from model.svm import MulticlassSVM

    print("\n[TRAINING] Support Vector Machine (SVM)")

    # user memilih jenis kernel yang akan digunakan
    # - linear: untuk data yang linearly separable
    # - rbf: untuk data non-linear (paling umum digunakan)
    # - polynomial: untuk data dengan polynomial decision boundary
    print("\nPilih kernel untuk SVM:")
    print("1. Linear")
    print("2. RBF (Radial Basis Function)")
    print("3. Polynomial")

    kernel_choice = input("\nPilihan kernel (1/2/3) [default: 2]: ").strip() or "2"

    # map pilihan user ke nama kernel
    kernel_map = {
        "1": "linear",
        "2": "rbf",
        "3": "poly"
    }
    kernel = kernel_map.get(kernel_choice, "rbf")

    print(f"\nMenggunakan kernel: {kernel}")

    start_time = time.time()
    # inisialisasi multiclass svm dengan parameter
    # untuk rbf/poly kernel, gunakan max_iter lebih tinggi untuk convergence
    max_iterations = 1000 if kernel in ['rbf', 'poly'] else 500
    model = MulticlassSVM(
        C=1.0,              # regularization parameter
        kernel=kernel,      # jenis kernel yang dipilih
        gamma='scale',      # kernel coefficient
        max_iter=max_iterations,  # maksimum iterasi training (1000 untuk rbf/poly, 500 untuk linear)
        strategy='ova',     # one-vs-all strategy
        verbose=True        # print progress
    )
    # train model
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    # evaluate pada validation set
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    # tampilkan hasil
    print(f"\nTraining complete ({train_time:.2f}s)")
    print(f"  Kernel: {kernel.upper()}")
    print(f"  Validation Accuracy: {accuracy:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_val, y_pred))

    return model, accuracy, train_time


def train_decision_tree(X_train, y_train, X_val, y_val):
    """train decision tree model"""
    from model.dtl import DecisionTree

    print("\n[TRAINING] Decision Tree")

    # user memilih criterion untuk splitting
    # - gini: mengukur impurity (lebih cepat, umumnya cukup baik)
    # - entropy: mengukur information gain (lebih slow tapi kadang lebih akurat)
    print("\nPilih criterion untuk Decision Tree:")
    print("1. Gini")
    print("2. Entropy")

    criterion_choice = input("\nPilihan criterion (1/2) [default: 1]: ").strip() or "1"
    criterion = "gini" if criterion_choice == "1" else "entropy"

    print(f"\nMenggunakan criterion: {criterion}")

    start_time = time.time()

    # inisialisasi decision tree dengan hyperparameter untuk prevent overfitting
    model = DecisionTree(
        min_samples_split=20,    # minimum samples untuk split node (pruning)
        max_depth=10,            # maximum depth tree (pruning)
        criterion=criterion      # splitting criterion yang dipilih user
    )

    # training bisa memakan waktu karena recursive tree building
    print("Training Decision Tree (ini mungkin memakan waktu)...")
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    # evaluate model performance
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    # tampilkan hasil training
    print(f"\nTraining complete ({train_time:.2f}s)")
    print(f"  Criterion: {criterion.upper()}")
    print(f"  Validation Accuracy: {accuracy:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_val, y_pred))

    # visualisasi tree (opsional)
    # user dapat memvisualisasikan struktur tree untuk memahami decision rules
    # visualisasi menampilkan nodes (decision rules) dan edges (true/false branches)
    # user juga bisa membatasi kedalaman yang ditampilkan (top-n levels)
    visualize = input("\nVisualisasikan struktur tree? (y/n) [default: n]: ").strip().lower() or "n"
    if visualize == 'y':
        print("\nParameter visualisasi:")
        # max_depth_vis = top-n levels yang akan ditampilkan
        # berguna jika tree terlalu besar dan sulit dibaca
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
    save trained model ke file .pkl
    model disimpan di current directory agar mudah diakses
    format pickle digunakan agar bisa di-load kembali untuk prediksi
    """
    save_choice = input("\nSimpan model? (y/n) [default: y]: ").strip().lower() or "y"

    if save_choice == 'y':
        try:
            filename = f"{model_name.lower().replace(' ', '_')}_model"
            # save to current directory (not storage/)
            model.save_model(filename, format='pkl', save_dir='.')
            print(f"\nModel berhasil disimpan sebagai: {filename}.pkl")
            print(f"   Lokasi: Current directory")
            return filename
        except Exception as e:
            print(f"\n[ERROR] Gagal menyimpan model: {e}")
            return None
    else:
        print("\nModel tidak disimpan.")
        return None


def save_predictions_to_csv(model, test_df, pipeline):
    """
    save predictions to csv file for kaggle submission
    fungsi ini akan generate file csv berisi prediksi untuk test data
    """
    from storage import ModelStorage

    save_csv = input("\nSimpan hasil dalam csv (y/n): ").strip().lower()

    if save_csv == 'y':
        csv_filename = input("Nama file CSV: ").strip()
        if not csv_filename:
            print("Nama file tidak boleh kosong!")
            return

        # tambahkan .csv extension jika belum ada
        if not csv_filename.endswith('.csv'):
            csv_filename += '.csv'

        try:
            # generate submission menggunakan model storage
            submission_df = ModelStorage.generate_submission(
                model=model,
                test_data=test_df,
                pipeline=pipeline,
                filename=csv_filename,
                id_column='Student_ID'
            )
            print(f"\n[OK] Prediksi berhasil disimpan ke: {csv_filename}")
        except Exception as e:
            print(f"\n[ERROR] Gagal menyimpan prediksi: {e}")


def train_single_sklearn_model(model_type, X_train, y_train, X_val, y_val, kernel='rbf'):
    """
    train single sklearn model untuk comparison
    """
    from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
    from sklearn.svm import SVC as SklearnSVM
    from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTree

    print(f"\n[SKLEARN] Training {model_type}...")
    start_time = time.time()

    if model_type == 'Logistic Regression':
        sklearn_model = SklearnLogisticRegression(
            max_iter=1000,
            penalty='l2',
            C=100,
            solver='lbfgs',
            multi_class='ovr',
            random_state=42
        )
    elif model_type == 'SVM':
        sklearn_model = SklearnSVM(
            C=1.0,
            kernel=kernel,
            gamma='scale',
            max_iter=1000,
            random_state=42
        )
    elif model_type == 'Decision Tree':
        sklearn_model = SklearnDecisionTree(
            min_samples_split=20,
            max_depth=10,
            criterion='gini',
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    sklearn_model.fit(X_train, y_train)
    train_time = time.time() - start_time
    predictions = sklearn_model.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)

    print(f"  accuracy: {accuracy:.4f}, time: {train_time:.2f}s")

    return sklearn_model, accuracy, train_time, predictions


def compare_single_model(scratch_model, scratch_pred, sklearn_model, sklearn_pred, y_val, model_name):
    """
    compare single model from scratch vs sklearn dan tampilkan hasil
    """
    print("\n" + "=" * 60)
    print(f"COMPARISON: {model_name.upper()}")
    print("=" * 60)

    # hitung metrics untuk from scratch
    scratch_metrics = {
        'accuracy': accuracy_score(y_val, scratch_pred),
        'precision': precision_score(y_val, scratch_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_val, scratch_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_val, scratch_pred, average='weighted', zero_division=0)
    }

    # hitung metrics untuk sklearn
    sklearn_metrics = {
        'accuracy': accuracy_score(y_val, sklearn_pred),
        'precision': precision_score(y_val, sklearn_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_val, sklearn_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_val, sklearn_pred, average='weighted', zero_division=0)
    }

    # tampilkan comparison
    print(f"\n{'Metric':<15} {'From Scratch':<15} {'Scikit-Learn':<15} {'Difference':<15}")
    print("-" * 60)
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        scratch_val = scratch_metrics[metric]
        sklearn_val = sklearn_metrics[metric]
        diff = scratch_val - sklearn_val
        print(f"{metric:<15} {scratch_val:<15.4f} {sklearn_val:<15.4f} {diff:+.4f}")
    print("-" * 60)

    # kesimpulan
    avg_diff = abs(sum(scratch_metrics[m] - sklearn_metrics[m] for m in scratch_metrics.keys()) / len(scratch_metrics))
    print(f"\nAverage absolute difference: {avg_diff:.4f}")

    if avg_diff < 0.02:
        print("[OK] EXCELLENT: From scratch performance sangat mendekati sklearn!")
    elif avg_diff < 0.05:
        print("[OK] GOOD: From scratch performance comparable dengan sklearn")
    else:
        print("[WARNING] GAP: Masih ada perbedaan performa yang cukup signifikan")


def train_sklearn_models(X_train, y_train, X_val, y_val):
    """
    train semua model dengan sklearn untuk comparison
    """
    from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
    from sklearn.svm import SVC as SklearnSVM
    from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTree

    sklearn_results = {}

    # train sklearn logistic regression
    print("\n[SKLEARN] Training Logistic Regression...")
    start_time = time.time()
    sklearn_logres = SklearnLogisticRegression(
        max_iter=1000,
        penalty='l2',
        C=100,  # c = 1/lambda_reg
        solver='lbfgs',
        multi_class='ovr',
        random_state=42
    )
    sklearn_logres.fit(X_train, y_train)
    logres_time = time.time() - start_time
    logres_pred = sklearn_logres.predict(X_val)
    logres_acc = accuracy_score(y_val, logres_pred)
    print(f"  accuracy: {logres_acc:.4f}, time: {logres_time:.2f}s")

    sklearn_results['Logistic Regression'] = {
        'model': sklearn_logres,
        'accuracy': logres_acc,
        'train_time': logres_time,
        'predictions': logres_pred
    }

    # train sklearn svm (rbf kernel)
    print("\n[SKLEARN] Training SVM (RBF kernel)...")
    start_time = time.time()
    sklearn_svm = SklearnSVM(
        C=1.0,
        kernel='rbf',
        gamma='scale',
        max_iter=1000,
        random_state=42
    )
    sklearn_svm.fit(X_train, y_train)
    svm_time = time.time() - start_time
    svm_pred = sklearn_svm.predict(X_val)
    svm_acc = accuracy_score(y_val, svm_pred)
    print(f"  accuracy: {svm_acc:.4f}, time: {svm_time:.2f}s")

    sklearn_results['SVM'] = {
        'model': sklearn_svm,
        'accuracy': svm_acc,
        'train_time': svm_time,
        'predictions': svm_pred
    }

    # train sklearn decision tree
    print("\n[SKLEARN] Training Decision Tree...")
    start_time = time.time()
    sklearn_dtl = SklearnDecisionTree(
        min_samples_split=20,
        max_depth=10,
        criterion='gini',
        random_state=42
    )
    sklearn_dtl.fit(X_train, y_train)
    dtl_time = time.time() - start_time
    dtl_pred = sklearn_dtl.predict(X_val)
    dtl_acc = accuracy_score(y_val, dtl_pred)
    print(f"  accuracy: {dtl_acc:.4f}, time: {dtl_time:.2f}s")

    sklearn_results['Decision Tree'] = {
        'model': sklearn_dtl,
        'accuracy': dtl_acc,
        'train_time': dtl_time,
        'predictions': dtl_pred
    }

    return sklearn_results


def generate_comparison_txt(scratch_results, sklearn_results, y_val, filename='model_comparison.txt'):
    """
    generate file txt berisi comparison lengkap antara from scratch vs sklearn
    """
    with open(filename, 'w', encoding='utf-8') as f:
        # header
        f.write("=" * 80 + "\n")
        f.write(" MODEL COMPARISON: FROM SCRATCH vs SCIKIT-LEARN\n")
        f.write("=" * 80 + "\n")
        f.write(f"generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        # summary table
        f.write("SUMMARY - VALIDATION ACCURACY\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Model':<25} {'From Scratch':<15} {'Scikit-Learn':<15} {'Difference':<15}\n")
        f.write("-" * 80 + "\n")

        for model_name in scratch_results.keys():
            scratch_acc = scratch_results[model_name]['accuracy']
            sklearn_acc = sklearn_results[model_name]['accuracy']
            diff = scratch_acc - sklearn_acc

            f.write(f"{model_name:<25} {scratch_acc:<15.4f} {sklearn_acc:<15.4f} {diff:+.4f}\n")

        f.write("-" * 80 + "\n\n")

        # detailed comparison untuk setiap model
        for model_name in scratch_results.keys():
            f.write("\n" + "=" * 80 + "\n")
            f.write(f" {model_name.upper()}\n")
            f.write("=" * 80 + "\n\n")

            # gunakan predictions yang sudah disimpan
            # predictions sudah di-compute dan disimpan saat training
            scratch_pred = scratch_results[model_name]['predictions']
            sklearn_pred = sklearn_results[model_name]['predictions']

            # hitung metrics lengkap
            scratch_metrics = {
                'accuracy': accuracy_score(y_val, scratch_pred),
                'precision': precision_score(y_val, scratch_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_val, scratch_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_val, scratch_pred, average='weighted', zero_division=0)
            }

            sklearn_metrics = {
                'accuracy': accuracy_score(y_val, sklearn_pred),
                'precision': precision_score(y_val, sklearn_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_val, sklearn_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_val, sklearn_pred, average='weighted', zero_division=0)
            }

            f.write("FROM SCRATCH:\n")
            f.write(f"  accuracy:  {scratch_metrics['accuracy']:.4f}\n")
            f.write(f"  precision: {scratch_metrics['precision']:.4f}\n")
            f.write(f"  recall:    {scratch_metrics['recall']:.4f}\n")
            f.write(f"  f1-score:  {scratch_metrics['f1']:.4f}\n")
            f.write(f"  time:      {scratch_results[model_name]['train_time']:.2f}s\n\n")

            f.write("SCIKIT-LEARN:\n")
            f.write(f"  accuracy:  {sklearn_metrics['accuracy']:.4f}\n")
            f.write(f"  precision: {sklearn_metrics['precision']:.4f}\n")
            f.write(f"  recall:    {sklearn_metrics['recall']:.4f}\n")
            f.write(f"  f1-score:  {sklearn_metrics['f1']:.4f}\n")
            f.write(f"  time:      {sklearn_results[model_name]['train_time']:.2f}s\n\n")

            f.write("DIFFERENCE (FROM SCRATCH - SCIKIT-LEARN):\n")
            f.write(f"  accuracy:  {scratch_metrics['accuracy'] - sklearn_metrics['accuracy']:+.4f}\n")
            f.write(f"  precision: {scratch_metrics['precision'] - sklearn_metrics['precision']:+.4f}\n")
            f.write(f"  recall:    {scratch_metrics['recall'] - sklearn_metrics['recall']:+.4f}\n")
            f.write(f"  f1-score:  {scratch_metrics['f1'] - sklearn_metrics['f1']:+.4f}\n")
            f.write(f"  time:      {scratch_results[model_name]['train_time'] - sklearn_results[model_name]['train_time']:+.2f}s\n")

        # footer
        f.write("\n" + "=" * 80 + "\n")
        f.write(" END OF COMPARISON\n")
        f.write("=" * 80 + "\n")

    print(f"\n[OK] Comparison report berhasil disimpan ke: {filename}")


def main():
    """main function dengan interactive menu"""

    # load and prepare data
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
        # train from scratch
        model, accuracy, train_time = train_logistic_regression(X_train, y_train, X_val, y_val)
        scratch_pred = model.predict(X_val)

        # comparison dengan sklearn
        run_comparison = input("\nJalankan comparison dengan scikit-learn? (y/n) [default: y]: ").strip().lower() or "y"
        if run_comparison == 'y':
            sklearn_model, sklearn_acc, sklearn_time, sklearn_pred = train_single_sklearn_model(
                'Logistic Regression', X_train, y_train, X_val, y_val
            )
            compare_single_model(model, scratch_pred, sklearn_model, sklearn_pred, y_val, 'Logistic Regression')

        save_model(model, "logistic_regression")

        # ask if user wants to save predictions to csv
        save_predictions_to_csv(model, test_df, pipeline)

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
        # train from scratch
        model, accuracy, train_time = train_svm(X_train, y_train, X_val, y_val)
        scratch_pred = model.predict(X_val)

        # comparison dengan sklearn
        run_comparison = input("\nJalankan comparison dengan scikit-learn? (y/n) [default: y]: ").strip().lower() or "y"
        if run_comparison == 'y':
            # detect kernel yang digunakan dari model scratch
            kernel_used = model.kernel if hasattr(model, 'kernel') else 'rbf'
            sklearn_model, sklearn_acc, sklearn_time, sklearn_pred = train_single_sklearn_model(
                'SVM', X_train, y_train, X_val, y_val, kernel=kernel_used
            )
            compare_single_model(model, scratch_pred, sklearn_model, sklearn_pred, y_val, f'SVM ({kernel_used.upper()})')

        save_model(model, "svm")

        # ask if user wants to save predictions to csv
        save_predictions_to_csv(model, test_df, pipeline)

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
        # train from scratch
        model, accuracy, train_time = train_decision_tree(X_train, y_train, X_val, y_val)
        scratch_pred = model.predict(X_val)

        # comparison dengan sklearn
        run_comparison = input("\nJalankan comparison dengan scikit-learn? (y/n) [default: y]: ").strip().lower() or "y"
        if run_comparison == 'y':
            sklearn_model, sklearn_acc, sklearn_time, sklearn_pred = train_single_sklearn_model(
                'Decision Tree', X_train, y_train, X_val, y_val
            )
            compare_single_model(model, scratch_pred, sklearn_model, sklearn_pred, y_val, 'Decision Tree')

        save_model(model, "decision_tree")

        # ask if user wants to save predictions to csv
        save_predictions_to_csv(model, test_df, pipeline)

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

        # train all models from scratch
        print("\nTraining 1/3: Logistic Regression")
        logres_model, logres_acc, logres_time = train_logistic_regression(X_train, y_train, X_val, y_val)
        logres_pred = logres_model.predict(X_val)  # simpan predictions
        models_results['Logistic Regression'] = {
            'model': logres_model,
            'accuracy': logres_acc,
            'train_time': logres_time,
            'predictions': logres_pred
        }

        # train model 2: svm dengan linear kernel
        # linear kernel bekerja sangat baik dengan naive sgd implementation
        # rbf kernel butuh qp solver (cvxopt) untuk optimal solution
        print("\nTraining 2/3: SVM (Linear kernel)")
        from model.svm import MulticlassSVM
        svm_model = MulticlassSVM(C=1.0, kernel='linear', max_iter=500, strategy='ova', verbose=True)
        start_time = time.time()
        svm_model.fit(X_train, y_train)
        svm_time = time.time() - start_time
        svm_pred = svm_model.predict(X_val)  # simpan predictions
        svm_acc = accuracy_score(y_val, svm_pred)
        print(f"\nSVM Accuracy: {svm_acc:.4f}, Time: {svm_time:.2f}s")
        models_results['SVM'] = {
            'model': svm_model,
            'accuracy': svm_acc,
            'train_time': svm_time,
            'predictions': svm_pred
        }

        # train model 3: decision tree dengan gini criterion (default)
        print("\nTraining 3/3: Decision Tree")
        from model.dtl import DecisionTree
        dtl_model = DecisionTree(min_samples_split=20, max_depth=10, criterion='gini')
        start_time = time.time()
        dtl_model.fit(X_train, y_train)
        dtl_time = time.time() - start_time
        dtl_pred = dtl_model.predict(X_val)  # simpan predictions
        dtl_acc = accuracy_score(y_val, dtl_pred)
        print(f"\nDecision Tree Accuracy: {dtl_acc:.4f}, Time: {dtl_time:.2f}s")
        models_results['Decision Tree'] = {
            'model': dtl_model,
            'accuracy': dtl_acc,
            'train_time': dtl_time,
            'predictions': dtl_pred
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

        # ========== SAVE ALL MODELS ==========
        # Simpan semua model yang sudah dilatih ke file .pkl
        print("\n" + "=" * 40)
        print("SAVING ALL MODELS")
        print("=" * 40)

        for model_name, results in models_results.items():
            try:
                filename = f"{model_name.lower().replace(' ', '_')}_model"
                results['model'].save_model(filename, format='pkl', save_dir='.')
                print(f"[OK] {model_name} disimpan sebagai: {filename}.pkl")
            except Exception as e:
                print(f"[ERROR] Gagal menyimpan {model_name}: {e}")

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

        # ========== KAGGLE SUBMISSION ==========
        # Generate submission CSV menggunakan model terbaik
        print("\n" + "=" * 40)
        print("KAGGLE SUBMISSION")
        print("=" * 40)
        print(f"\nModel terbaik untuk submit ke Kaggle: {best_model_name}")
        print(f"  Validation Accuracy: {best_accuracy:.4f}")

        generate_submission = input("\nGenerate submission file untuk Kaggle? (y/n) [default: y]: ").strip().lower() or "y"

        if generate_submission == 'y':
            from storage import ModelStorage

            csv_filename = input("Nama file submission [default: submission.csv]: ").strip() or "submission.csv"
            if not csv_filename.endswith('.csv'):
                csv_filename += '.csv'

            try:
                # Generate submission menggunakan model terbaik
                submission_df = ModelStorage.generate_submission(
                    model=models_results[best_model_name]['model'],
                    test_data=test_df,
                    pipeline=pipeline,
                    filename=csv_filename,
                    id_column='Student_ID'
                )
                print(f"\n[OK] Submission berhasil dibuat: {csv_filename}")
                print(f"  Ready untuk di-upload ke Kaggle!")
            except Exception as e:
                print(f"\n[ERROR] Gagal membuat submission: {e}")

        # ========== COMPARISON FROM SCRATCH vs SKLEARN ==========
        # train sklearn models untuk comparison
        print("\n" + "=" * 40)
        print("COMPARISON: FROM SCRATCH vs SKLEARN")
        print("=" * 40)

        run_comparison = input("\nJalankan comparison dengan scikit-learn? (y/n) [default: y]: ").strip().lower() or "y"

        if run_comparison == 'y':
            # train sklearn models
            sklearn_results = train_sklearn_models(X_train, y_train, X_val, y_val)

            # generate comparison txt file
            comparison_filename = input("\nNama file output comparison [default: model_comparison.txt]: ").strip() or "model_comparison.txt"
            if not comparison_filename.endswith('.txt'):
                comparison_filename += '.txt'

            generate_comparison_txt(models_results, sklearn_results, y_val, comparison_filename)

            # tampilkan summary comparison
            print("\n" + "=" * 40)
            print("COMPARISON SUMMARY")
            print("=" * 40)
            print(f"{'Model':<25} {'From Scratch':<15} {'Scikit-Learn':<15} {'Difference':<15}")
            print("-" * 40)

            for model_name in models_results.keys():
                scratch_acc = models_results[model_name]['accuracy']
                sklearn_acc = sklearn_results[model_name]['accuracy']
                diff = scratch_acc - sklearn_acc

                print(f"{model_name:<25} {scratch_acc:<15.4f} {sklearn_acc:<15.4f} {diff:+.4f}")

            print("-" * 40)
            print(f"\n[OK] Comparison selesai! Lihat detail di: {comparison_filename}")

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
