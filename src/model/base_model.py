from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict, Literal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from storage.model_storage import save_model as storage_save, load_model as storage_load

class BaseModel(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseModel':
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        predictions = self.predict(X)
        return np.mean(predictions == y)

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        pass

    def save_model(self,
                   filename: str,
                   format: Literal['pkl', 'txt', 'both'] = 'both',
                   save_dir: str = None) -> None:
        storage_save(self, filename, format, save_dir)

    @classmethod
    def load_model(cls,
                   filename: str,
                   format: Literal['pkl', 'txt'] = 'pkl',
                   load_dir: str = None) -> 'BaseModel':
        return storage_load(cls, filename, format, load_dir)


class BinaryClassifierMixin:
    @staticmethod
    def convert_labels(y: np.ndarray) -> np.ndarray:
        return np.where(y <= 0, -1, 1)


class MulticlassClassifierMixin:
    def _initialize_multiclass(self, y: np.ndarray) -> None:
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)

        if self.n_classes < 2:
            raise ValueError("jumlah kelas harus minimal 2")
