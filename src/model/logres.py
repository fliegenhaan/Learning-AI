"""
Logistic Regression Implementation
Implementasi Logistic Regression dari nol menggunakan gradient descent
"""

import numpy as np
from typing import Optional, Tuple, List
from .base_model import BaseModel, BinaryClassifierMixin, MulticlassClassifierMixin


class LogisticRegression(BaseModel, BinaryClassifierMixin):
    """binary logistic regression classifier menggunakan gradient descent"""

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        regularization: Optional[str] = None,
        lambda_reg: float = 0.01,
        verbose: bool = False,
        tol: float = 1e-4
    ):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.verbose = verbose
        self.tol = tol

        self.weights = None
        self.bias = None
        self.loss_history = []
        self.accuracy_history = []

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """sigmoid activation function"""
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """compute binary cross-entropy loss dengan regularization"""
        m = len(y_true)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

        if self.regularization == 'l2':
            loss += (self.lambda_reg / (2 * m)) * np.sum(self.weights ** 2)
        elif self.regularization == 'l1':
            loss += (self.lambda_reg / m) * np.sum(np.abs(self.weights))

        return loss

    def _compute_gradients(self, X: np.ndarray, y_true: np.ndarray,
                          y_pred: np.ndarray) -> Tuple[np.ndarray, float]:
        """compute gradients untuk weights dan bias"""
        m = len(y_true)
        error = y_pred - y_true

        dw = (1 / m) * np.dot(X.T, error)

        if self.regularization == 'l2':
            dw += (self.lambda_reg / m) * self.weights
        elif self.regularization == 'l1':
            dw += (self.lambda_reg / m) * np.sign(self.weights)

        db = (1 / m) * np.sum(error)

        return dw, db

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegression':
        """melatih model logistic regression"""
        # convert pandas to numpy jika perlu
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values

        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape

        if self.verbose:
            print(f"  jumlah sampel: {n_samples}, jumlah fitur: {n_features}")
            print(f"  learning rate: {self.learning_rate}, iterations: {self.n_iterations}")

        self.weights = np.zeros(n_features)
        self.bias = 0
        self.loss_history = []
        self.accuracy_history = []

        for iteration in range(self.n_iterations):
            linear_output = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_output)
            loss = self._compute_loss(y, y_pred)
            self.loss_history.append(loss)
            y_pred_class = (y_pred >= 0.5).astype(int)
            accuracy = np.mean(y_pred_class == y)
            self.accuracy_history.append(accuracy)
            dw, db = self._compute_gradients(X, y, y_pred)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if self.verbose and (iteration % 100 == 0 or iteration == self.n_iterations - 1):
                print(f"  iterasi {iteration}: loss = {loss:.4f}, accuracy = {accuracy:.4f}")

            if iteration > 0 and abs(self.loss_history[-1] - self.loss_history[-2]) < self.tol:
                if self.verbose:
                    print(f"  early stopping di iterasi {iteration}")
                break

        if self.verbose:
            print(f"  training selesai!")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """predict probability untuk setiap sample"""
        if self.weights is None or self.bias is None:
            raise ValueError("model belum di-fit, jalankan fit() terlebih dahulu")

        if hasattr(X, 'values'):
            X = X.values
        X = np.array(X)

        linear_output = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_output)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """predict class labels"""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

    def get_params(self) -> dict:
        """mendapatkan parameter model"""
        return {
            'weights': self.weights,
            'bias': self.bias,
            'learning_rate': self.learning_rate,
            'n_iterations': self.n_iterations,
            'regularization': self.regularization,
            'lambda_reg': self.lambda_reg,
            'n_features': len(self.weights) if self.weights is not None else 0
        }

    def get_loss_history(self) -> List[float]:
        """get training loss history"""
        return self.loss_history

    def get_accuracy_history(self) -> List[float]:
        """get training accuracy history"""
        return self.accuracy_history


class MultiClassLogisticRegression(BaseModel, MulticlassClassifierMixin):
    """multiclass logistic regression menggunakan strategi One-vs-Rest"""

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        regularization: Optional[str] = None,
        lambda_reg: float = 0.01,
        verbose: bool = False
    ):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.verbose = verbose

        self.models = {}
        self.classes = None
        self.n_classes = None

    def _create_classifier(self) -> LogisticRegression:
        """membuat instance binary logistic regression classifier"""
        return LogisticRegression(
            learning_rate=self.learning_rate,
            n_iterations=self.n_iterations,
            regularization=self.regularization,
            lambda_reg=self.lambda_reg,
            verbose=self.verbose
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MultiClassLogisticRegression':
        """melatih multiclass logistic regression"""
        # convert pandas to numpy jika perlu
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        X = np.array(X)
        y = np.array(y)
        self._initialize_multiclass(y)
        if self.verbose:
            print(f"melatih {self.n_classes} binary classifiers dengan strategi One-vs-Rest")
        for i, cls in enumerate(self.classes):
            if self.verbose:
                print(f"\ntraining classifier {i+1}/{self.n_classes} untuk kelas {cls}")
            y_binary = (y == cls).astype(int)
            model = self._create_classifier()
            model.fit(X, y_binary)
            self.models[cls] = model
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """predict probabilities untuk setiap class"""
        if hasattr(X, 'values'):
            X = X.values
        X = np.array(X)
        n_samples = X.shape[0]
        proba = np.zeros((n_samples, self.n_classes))
        for i, cls in enumerate(self.classes):
            proba[:, i] = self.models[cls].predict_proba(X)
        proba_sum = proba.sum(axis=1, keepdims=True)
        proba = proba / proba_sum
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        """predict class labels"""
        proba = self.predict_proba(X)
        class_indices = np.argmax(proba, axis=1)
        return self.classes[class_indices]

    def get_params(self) -> dict:
        """mendapatkan parameter model"""
        return {
            'learning_rate': self.learning_rate,
            'n_iterations': self.n_iterations,
            'regularization': self.regularization,
            'lambda_reg': self.lambda_reg,
            'n_classes': self.n_classes,
            'n_classifiers': len(self.models)
        }
