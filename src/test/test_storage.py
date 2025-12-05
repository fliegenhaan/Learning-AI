import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from model.svm import SVM
from storage import list_saved_models

def create_sample_data():
    np.random.seed(42)
    X_class1 = np.random.randn(50, 2) + np.array([2, 2])
    X_class2 = np.random.randn(50, 2) + np.array([-2, -2])
    X = np.vstack([X_class1, X_class2])
    y = np.array([1] * 50 + [-1] * 50)
    return X, y

def test_save_load():
    print("=" * 60)
    print("Test Save & Load Model")
    print("=" * 60)

    X_train, y_train = create_sample_data()

    print("\n1. Training SVM model")
    svm = SVM(C=1.0, kernel='linear', max_iter=100, verbose=False)
    svm.fit(X_train, y_train)
    accuracy_original = svm.score(X_train, y_train)
    print(f"Akurasi: {accuracy_original:.2%}")

    print("\n2. Save model (PKL & TXT)")
    svm.save_model('test_svm_storage', format='both')

    print("\n3. Load dari PKL")
    svm_from_pkl = SVM.load_model('test_svm_storage', format='pkl')
    accuracy_pkl = svm_from_pkl.score(X_train, y_train)
    print(f"Akurasi PKL: {accuracy_pkl:.2%}")

    print("\n4. Load dari TXT")
    svm_from_txt = SVM.load_model('test_svm_storage', format='txt')
    accuracy_txt = svm_from_txt.score(X_train, y_train)
    print(f"Akurasi TXT: {accuracy_txt:.2%}")

    print("\n5. Verifikasi prediksi")
    pred_original = svm.predict(X_train[:5])
    pred_pkl = svm_from_pkl.predict(X_train[:5])
    pred_txt = svm_from_txt.predict(X_train[:5])

    print(f"Original: {pred_original}")
    print(f"PKL:      {pred_pkl}")
    print(f"TXT:      {pred_txt}")

    if np.allclose(pred_original, pred_pkl) and np.allclose(pred_original, pred_txt):
        print("Semua prediksi sama")
    else:
        print("Prediksi tidak sama")

    print("\n6. List model tersimpan")
    models = list_saved_models()
    print(f"Model: {models}")

    print(f"\nRingkasan:")
    print(f"- Akurasi original: {accuracy_original:.2%}")
    print(f"- Akurasi PKL: {accuracy_pkl:.2%}")
    print(f"- Akurasi TXT: {accuracy_txt:.2%}")
    print(f"- Lokasi: src/storage/")

if __name__ == '__main__':
    test_save_load()