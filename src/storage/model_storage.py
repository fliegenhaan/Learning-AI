import os
import pickle
import json
import numpy as np
import pandas as pd
import glob
from typing import Any, Dict, Literal, Tuple
from pathlib import Path


class ModelStorage:
    DEFAULT_DIR = Path(__file__).parent

    @staticmethod
    def save(model: Any,
             filename: str,
             format: Literal['pkl', 'txt', 'both'] = 'both',
             save_dir: str = None) -> None:
        if save_dir is None:
            save_dir = ModelStorage.DEFAULT_DIR
        else:
            save_dir = Path(save_dir)

        save_dir.mkdir(parents=True, exist_ok=True)

        if format in ['pkl', 'both']:
            pkl_path = save_dir / f"{filename}.pkl"
            ModelStorage._save_pkl(model, pkl_path)
            print(f"[SAVE] Model disimpan (PKL): {pkl_path}")

        if format in ['txt', 'both']:
            txt_path = save_dir / f"{filename}.txt"
            ModelStorage._save_txt(model, txt_path)
            print(f"[SAVE] Model disimpan (TXT): {txt_path}")

    @staticmethod
    def _save_pkl(model: Any, filepath: Path) -> None:
        with open(filepath, 'wb') as f:
            pickle.dump(model.__dict__, f)

    @staticmethod
    def _save_txt(model: Any, filepath: Path) -> None:
        model_data = {
            'model_type': model.__class__.__name__,
            'model_module': model.__class__.__module__,
            'parameters': {},
            'arrays': {}
        }

        for key, value in model.__dict__.items():
            if isinstance(value, np.ndarray):
                model_data['arrays'][key] = {
                    'shape': list(value.shape),
                    'dtype': str(value.dtype),
                    'data': value.tolist()
                }
            elif isinstance(value, (int, float, str, bool, type(None))):
                model_data['parameters'][key] = value
            elif isinstance(value, (list, tuple)):
                model_data['parameters'][key] = ModelStorage._convert_to_serializable(value)
            elif hasattr(value, '__dict__'):
                try:
                    model_data['parameters'][key] = str(value)
                except:
                    model_data['parameters'][key] = f"<{type(value).__name__} object>"
            else:
                model_data['parameters'][key] = str(value)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return [ModelStorage._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: ModelStorage._convert_to_serializable(v) for k, v in obj.items()}
        else:
            return obj

    @staticmethod
    def load(model_class: type,
             filename: str,
             format: Literal['pkl', 'txt'] = 'pkl',
             load_dir: str = None) -> Any:
        if load_dir is None:
            load_dir = ModelStorage.DEFAULT_DIR
        else:
            load_dir = Path(load_dir)

        if format == 'pkl':
            pkl_path = load_dir / f"{filename}.pkl"
            model = ModelStorage._load_pkl(model_class, pkl_path)
            print(f"[LOAD] Model dimuat (PKL): {pkl_path}")
        elif format == 'txt':
            txt_path = load_dir / f"{filename}.txt"
            model = ModelStorage._load_txt(model_class, txt_path)
            print(f"[LOAD] Model dimuat (TXT): {txt_path}")
        else:
            raise ValueError(f"Format '{format}' tidak dikenali")

        return model

    @staticmethod
    def _load_pkl(model_class: type, filepath: Path) -> Any:
        if not filepath.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {filepath}")

        with open(filepath, 'rb') as f:
            model_dict = pickle.load(f)

        model = model_class.__new__(model_class)
        model.__dict__.update(model_dict)
        return model

    @staticmethod
    def _load_txt(model_class: type, filepath: Path) -> Any:
        if not filepath.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            model_data = json.load(f)

        model = model_class.__new__(model_class)

        for key, value in model_data['parameters'].items():
            setattr(model, key, value)

        for key, array_data in model_data['arrays'].items():
            arr = np.array(array_data['data'], dtype=array_data['dtype'])
            arr = arr.reshape(array_data['shape'])
            setattr(model, key, arr)

        return model

    @staticmethod
    def list_models(directory: str = None, format: Literal['pkl', 'txt', 'all'] = 'all') -> list:
        if directory is None:
            directory = ModelStorage.DEFAULT_DIR
        else:
            directory = Path(directory)

        if not directory.exists():
            return []

        models = []

        if format in ['pkl', 'all']:
            pkl_files = list(directory.glob('*.pkl'))
            models.extend([f.stem for f in pkl_files])

        if format in ['txt', 'all']:
            txt_files = list(directory.glob('*.txt'))
            models.extend([f.stem for f in txt_files])

        return sorted(list(set(models)))

    @staticmethod
    def load_model_for_prediction(model_path: str) -> Any:
        """Load model untuk prediksi (auto-detect path)"""
        model_path = Path(model_path)

        if not model_path.exists():
            model_path = Path(str(model_path) + '.pkl')

        if not model_path.exists():
            raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")

        print(f"Loading model from: {model_path}")

        with open(model_path, 'rb') as f:
            model_dict = pickle.load(f)

        # Reconstruct model dari dict
        # Detect model class from dict
        if 'classifiers' in model_dict:
            # MulticlassSVM or MultiClassLogisticRegression
            if 'kernel' in model_dict:
                from model.svm import MulticlassSVM
                model = MulticlassSVM.__new__(MulticlassSVM)
            else:
                from model.logres import MultiClassLogisticRegression
                model = MultiClassLogisticRegression.__new__(MultiClassLogisticRegression)
        elif 'root' in model_dict:
            # DecisionTree
            from model.dtl import DecisionTree
            model = DecisionTree.__new__(DecisionTree)
        elif 'models' in model_dict:
            # MultiClassLogisticRegression
            from model.logres import MultiClassLogisticRegression
            model = MultiClassLogisticRegression.__new__(MultiClassLogisticRegression)
        elif 'w' in model_dict or 'alpha' in model_dict:
            # SVM
            from model.svm import SVM
            model = SVM.__new__(SVM)
        elif 'weights' in model_dict:
            # LogisticRegression
            from model.logres import LogisticRegression
            model = LogisticRegression.__new__(LogisticRegression)
        else:
            raise ValueError("Cannot determine model type from saved data")

        model.__dict__.update(model_dict)
        print(f"Model loaded: {model.__class__.__name__}")
        return model

    @staticmethod
    def generate_submission(
        model: Any,
        test_data: pd.DataFrame,
        pipeline: Any,
        filename: str = "submission.csv",
        id_column: str = 'id'
    ) -> pd.DataFrame:
        """Generate submission file untuk Kaggle"""
        print("\n[GENERATING SUBMISSION]")

        # Extract IDs
        if id_column in test_data.columns:
            test_ids = test_data[id_column]
            X_test = test_data.drop(id_column, axis=1)
        else:
            test_ids = test_data.index
            X_test = test_data.copy()

        print(f"Test data shape: {X_test.shape}")

        # Transform test data
        print("Preprocessing test data...")
        X_test_transformed = pipeline.transform(X_test)
        print(f"Transformed shape: {X_test_transformed.shape}")

        # Predict
        print("Making predictions...")
        predictions = model.predict(X_test_transformed)

        # Create submission DataFrame
        submission_df = pd.DataFrame({
            id_column: test_ids,
            'Target': predictions
        })

        # Save to CSV
        submission_df.to_csv(filename, index=False)

        print(f"\nSubmission saved: {filename}")
        print(f"  Total predictions: {len(predictions)}")
        print(f"\nPrediction distribution:")
        print(submission_df['Target'].value_counts())

        return submission_df


def save_model(model: Any,
               filename: str,
               format: Literal['pkl', 'txt', 'both'] = 'both',
               save_dir: str = None) -> None:
    ModelStorage.save(model, filename, format, save_dir)


def load_model(model_class: type,
               filename: str,
               format: Literal['pkl', 'txt'] = 'pkl',
               load_dir: str = None) -> Any:
    return ModelStorage.load(model_class, filename, format, load_dir)


def list_saved_models(directory: str = None, format: Literal['pkl', 'txt', 'all'] = 'all') -> list:
    return ModelStorage.list_models(directory, format)
