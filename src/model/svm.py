import numpy as np
from typing import Literal, Union
from .base_model import BaseModel, BinaryClassifierMixin, MulticlassClassifierMixin


class SVM(BaseModel, BinaryClassifierMixin):
    """binary svm classifier menggunakan naive sgd"""

    def __init__(
        self,
        C: float = 1.0,
        kernel: Literal['linear', 'poly', 'rbf'] = 'linear',
        degree: int = 3,
        gamma: Union[float, Literal['auto', 'scale'], None] = None,
        coef0: float = 0.0,
        tol: float = 1e-3,
        max_iter: int = 1000,
        learning_rate: Literal['optimal', 'constant', 'adaptive'] = 'optimal',
        eta0: float = 0.01,
        verbose: bool = False
    ):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.verbose = verbose

        self.w = None
        self.b = None
        self.alpha = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.iteration_count = 0

    def _kernel_function(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """menghitung kernel matrix antara X1 dan X2 (fully vectorized)"""
        if hasattr(X1, 'values'):
            X1 = X1.values
        if hasattr(X2, 'values'):
            X2 = X2.values

        if self.kernel == 'linear':
            return np.dot(X1, X2.T)

        elif self.kernel == 'poly':
            gamma = self.gamma if self.gamma is not None else 1.0 / X1.shape[1]
            return (gamma * np.dot(X1, X2.T) + self.coef0) ** self.degree

        elif self.kernel == 'rbf':
            gamma = self.gamma if self.gamma is not None else 1.0 / X1.shape[1]
            X1_norm = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
            X2_norm = np.sum(X2 ** 2, axis=1).reshape(1, -1)
            distances = X1_norm + X2_norm - 2 * np.dot(X1, X2.T)
            return np.exp(-gamma * distances)

        raise ValueError(f"kernel '{self.kernel}' tidak dikenali")

    def _compute_learning_rate(self, t: int, lambda_reg: float) -> float:
        """menghitung learning rate berdasarkan strategi"""
        if self.learning_rate == 'optimal':
            return 1.0 / (lambda_reg * t)
        elif self.learning_rate == 'constant':
            return self.eta0
        elif self.learning_rate == 'adaptive':
            return self.eta0 / np.sqrt(t)
        return self.eta0

    def _compute_hinge_loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> float:
        """menghitung hinge loss (fully vectorized)"""
        margins = y * (np.dot(X, w) + b)
        losses = np.maximum(0, 1 - margins)
        return np.mean(losses)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVM':
        """melatih svm dengan naive sgd"""
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values

        n_samples, n_features = X.shape
        y = self.convert_labels(y)

        if self.verbose:
            print(f"  jumlah sampel: {n_samples}, jumlah fitur: {n_features}")
            print(f"  kernel: {self.kernel}, C: {self.C}, max_iter: {self.max_iter}")

        if self.gamma is None or self.gamma == 'auto':
            self.gamma = 1.0 / n_features
        elif self.gamma == 'scale':
            self.gamma = 1.0 / (n_features * np.var(X))

        lambda_reg = 1.0 / (self.C * n_samples)

        if self.kernel == 'linear':
            self._fit_linear(X, y, lambda_reg, n_samples, n_features)
        else:
            self._fit_kernel(X, y, lambda_reg, n_samples)

        if self.verbose:
            print(f"  training selesai!")

        return self

    def _fit_linear(self, X: np.ndarray, y: np.ndarray, lambda_reg: float,
                    n_samples: int, n_features: int) -> None:
        """naive sgd untuk kernel linear"""
        self.w = np.zeros(n_features)
        self.b = 0.0

        if self.verbose:
            print(f"  samples: {n_samples}, will process {self.max_iter * n_samples} updates")

        for epoch in range(self.max_iter):
            indices = np.random.permutation(n_samples)
            epoch_loss = 0.0

            for i in indices:
                t = epoch * n_samples + i + 1

                xi = X[i]
                yi = y[i]

                if self.learning_rate == 'optimal':
                    eta = 1.0 / (lambda_reg * t)
                elif self.learning_rate == 'constant':
                    eta = self.eta0
                else:
                    eta = self.eta0 / np.sqrt(t)

                margin = yi * (np.dot(xi, self.w) + self.b)

                if margin < 1:
                    self.w = (1 - eta * lambda_reg) * self.w + eta * yi * xi
                    self.b = self.b + eta * yi
                else:
                    self.w = (1 - eta * lambda_reg) * self.w

                epoch_loss += max(0, 1 - margin)

            if self.verbose and (epoch + 1) % 100 == 0:
                avg_loss = epoch_loss / n_samples
                margins_all = y * (np.dot(X, self.w) + self.b)
                acc = np.mean(margins_all > 0)
                print(f"  epoch {epoch+1}/{self.max_iter}, avg_loss: {avg_loss:.6f}, acc: {acc:.4f}")

        margins = y * (np.dot(X, self.w) + self.b)
        sv_mask = margins <= 1 + self.tol
        self.support_vectors = X[sv_mask]
        self.support_vector_labels = y[sv_mask]

        if self.verbose:
            print(f"  jumlah support vectors: {np.sum(sv_mask)} ({np.sum(sv_mask)/n_samples*100:.1f}%)")

    def _fit_kernel(self, X: np.ndarray, y: np.ndarray, lambda_reg: float, n_samples: int) -> None:
        """naive sgd untuk kernel non-linear (rbf, poly)"""
        self.alpha = np.zeros(n_samples)
        self.support_vectors = X.copy()
        self.support_vector_labels = y.copy()
        self.b = 0.0

        for epoch in range(self.max_iter):
            indices = np.random.permutation(n_samples)
            epoch_loss = 0.0

            for i in indices:
                t = epoch * n_samples + i + 1

                if self.learning_rate == 'optimal':
                    eta = 1.0 / (lambda_reg * t)
                elif self.learning_rate == 'constant':
                    eta = self.eta0
                else:
                    eta = self.eta0 / np.sqrt(t)

                K_i = self._kernel_function(X[i:i+1], X).flatten()
                f_i = np.dot(K_i, self.alpha * self.support_vector_labels) + self.b
                margin = y[i] * f_i

                self.alpha *= (1 - eta * lambda_reg)

                if margin < 1:
                    self.alpha[i] += eta * y[i]
                    self.b += eta * y[i]

                epoch_loss += max(0, 1 - margin)

            if self.verbose and (epoch + 1) % 100 == 0:
                avg_loss = epoch_loss / n_samples
                print(f"  epoch {epoch+1}/{self.max_iter}, avg_loss: {avg_loss:.6f}")

        sv_indices = self.alpha > self.tol
        self.alpha = self.alpha[sv_indices]
        self.support_vectors = X[sv_indices]
        self.support_vector_labels = y[sv_indices]

        if self.verbose:
            print(f"  jumlah support vectors: {len(self.support_vectors)} ({len(self.support_vectors)/n_samples*100:.1f}%)")

    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        """hitung decision function f(x) = w·x + b"""
        if self.kernel == 'linear' and self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            K = self._kernel_function(X, self.support_vectors)
            return np.dot(K, self.alpha * self.support_vector_labels) + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """prediksi label untuk data baru"""
        return np.sign(self._decision_function(X))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """menghitung akurasi model"""
        y = self.convert_labels(y)
        return np.mean(self.predict(X) == y)

    def get_params(self) -> dict:
        """mengambil parameter model"""
        params = {
            'C': self.C,
            'kernel': self.kernel,
            'degree': self.degree,
            'gamma': self.gamma,
            'coef0': self.coef0,
            'tol': self.tol,
            'max_iter': self.max_iter,
            'learning_rate': self.learning_rate,
            'algorithm': 'Naive SGD',
            'n_support_vectors': len(self.support_vectors) if self.support_vectors is not None else 0
        }

        if self.kernel == 'linear' and self.w is not None:
            params['weight_norm'] = float(np.linalg.norm(self.w))

        return params


class MulticlassSVM(BaseModel, MulticlassClassifierMixin):
    """multiclass svm dengan strategi one-vs-all atau one-vs-one"""

    def __init__(
        self,
        C: float = 1.0,
        kernel: Literal['linear', 'poly', 'rbf'] = 'linear',
        degree: int = 3,
        gamma: Union[float, Literal['auto', 'scale'], None] = None,
        coef0: float = 0.0,
        tol: float = 1e-3,
        max_iter: int = 1000,
        learning_rate: Literal['optimal', 'constant', 'adaptive'] = 'optimal',
        eta0: float = 0.01,
        strategy: Literal['ova', 'ovo'] = 'ova',
        verbose: bool = False
    ):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.strategy = strategy
        self.verbose = verbose

        self.classifiers = []
        self.classes = None
        self.n_classes = None

    def _create_classifier(self) -> SVM:
        """buat instance binary svm classifier"""
        return SVM(
            C=self.C,
            kernel=self.kernel,
            degree=self.degree,
            gamma=self.gamma,
            coef0=self.coef0,
            tol=self.tol,
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            eta0=self.eta0,
            verbose=self.verbose
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MulticlassSVM':
        """melatih multiclass svm dengan strategi ova atau ovo"""
        self._initialize_multiclass(y)

        if self.n_classes == 2:
            print("hanya 2 kelas terdeteksi, menggunakan binary SVM")
            clf = self._create_classifier()
            y_binary = np.where(y == self.classes[0], -1, 1)
            clf.fit(X, y_binary)
            self.classifiers.append(clf)

        elif self.strategy == 'ova':
            self._fit_ova(X, y)

        elif self.strategy == 'ovo':
            self._fit_ovo(X, y)

        else:
            raise ValueError(f"strategy '{self.strategy}' tidak dikenali")

        return self

    def _fit_ova(self, X: np.ndarray, y: np.ndarray) -> None:
        """melatih dengan strategi one-vs-all"""
        print(f"melatih {self.n_classes} binary classifiers dengan strategi One-vs-All")

        for i, cls in enumerate(self.classes):
            print(f"training classifier {i+1}/{self.n_classes} untuk kelas {cls}")
            y_binary = np.where(y == cls, 1, -1)
            clf = self._create_classifier()
            clf.fit(X, y_binary)
            self.classifiers.append(clf)

    def _fit_ovo(self, X: np.ndarray, y: np.ndarray) -> None:
        """melatih dengan strategi one-vs-one"""
        n_classifiers = self.n_classes * (self.n_classes - 1) // 2
        print(f"melatih {n_classifiers} binary classifiers dengan strategi One-vs-One")

        classifier_idx = 0
        for i in range(self.n_classes):
            for j in range(i + 1, self.n_classes):
                classifier_idx += 1
                print(f"training classifier {classifier_idx}/{n_classifiers} untuk kelas {self.classes[i]} vs {self.classes[j]}")

                mask = (y == self.classes[i]) | (y == self.classes[j])
                X_pair = X[mask]
                y_pair = y[mask]
                y_binary = np.where(y_pair == self.classes[i], 1, -1)

                clf = self._create_classifier()
                clf.fit(X_pair, y_binary)
                self.classifiers.append((clf, self.classes[i], self.classes[j]))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """prediksi label untuk data baru"""
        n_samples = X.shape[0]

        if self.n_classes == 2:
            predictions = self.classifiers[0].predict(X)
            return np.where(predictions == -1, self.classes[0], self.classes[1])

        elif self.strategy == 'ova':
            return self._predict_ova(X, n_samples)

        elif self.strategy == 'ovo':
            return self._predict_ovo(X, n_samples)

    def _predict_ova(self, X: np.ndarray, n_samples: int) -> np.ndarray:
        """prediksi dengan strategi one-vs-all"""
        decision_scores = np.zeros((n_samples, self.n_classes))

        for i, clf in enumerate(self.classifiers):
            decision_scores[:, i] = clf._decision_function(X)

        class_indices = np.argmax(decision_scores, axis=1)
        return self.classes[class_indices]

    def _predict_ovo(self, X: np.ndarray, n_samples: int) -> np.ndarray:
        """prediksi dengan strategi one-vs-one menggunakan voting"""
        votes = np.zeros((n_samples, self.n_classes))

        for clf, class_i, class_j in self.classifiers:
            predictions = clf.predict(X)

            for sample_idx in range(n_samples):
                if predictions[sample_idx] == 1:
                    class_idx = np.where(self.classes == class_i)[0][0]
                    votes[sample_idx, class_idx] += 1
                else:
                    class_idx = np.where(self.classes == class_j)[0][0]
                    votes[sample_idx, class_idx] += 1

        class_indices = np.argmax(votes, axis=1)
        return self.classes[class_indices]

    def get_params(self) -> dict:
        """mendapatkan parameter model"""
        return {
            'C': self.C,
            'kernel': self.kernel,
            'degree': self.degree,
            'gamma': self.gamma,
            'coef0': self.coef0,
            'tol': self.tol,
            'max_iter': self.max_iter,
            'learning_rate': self.learning_rate,
            'strategy': self.strategy,
            'algorithm': 'Naive SGD',
            'n_classes': self.n_classes,
            'n_classifiers': len(self.classifiers)
        }
