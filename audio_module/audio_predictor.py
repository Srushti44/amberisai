"""
audio_predictor.py
==================
Production inference engine for AmberisAI infant cry classification.

Architecture:
  - Singleton pattern: models loaded once, reused across requests (Flask-ready).
  - Accepts file paths or raw numpy arrays.
  - Returns structured JSON-compatible output with calibrated probabilities.
  - Ensemble: 0.7 * RandomForest + 0.3 * XGBoost soft probability fusion.

Output format:
    {
        "module": "audio",
        "detected_condition": "hungry",
        "confidence": 0.84,
        "secondary_condition": "discomfort",
        "all_probabilities": {
            "hungry": 0.84,
            "discomfort": 0.09,
            "belly_pain": 0.04,
            "tired": 0.03,
            "burping": 0.00
        },
        "meta": {
            "model": "ensemble_rf_xgb",
            "ensemble_weights": {"rf": 0.7, "xgb": 0.3},
            "feature_dim": 154,
            "audio_duration_seconds": 3.8
        }
    }

Confidence design:
  - Derived from soft probabilities (NOT argmax alone).
  - Reflects genuine model uncertainty: low confidence → spread probabilities.
  - Never forces a confident output on ambiguous input.
"""

import os
import pickle
import logging
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Union

import numpy as np

from .feature_extraction import FeatureExtractor
from .utils import load_audio, validate_audio_file, load_audio_with_checks

logger = logging.getLogger(__name__)

# ─── ARTIFACT FILENAMES (must match train_audio_model.py) ─────────────────────
RF_MODEL_FILE = "random_forest.pkl"
XGB_MODEL_FILE = "xgboost.pkl"
SCALER_FILE = "scaler.pkl"
LABEL_ENCODER_FILE = "label_encoder.pkl"
FEATURE_NAMES_FILE = "feature_names.pkl"
TRAINING_META_FILE = "training_meta.pkl"

# Ensemble weights
RF_WEIGHT = 0.7
XGB_WEIGHT = 0.3

# Confidence threshold below which we flag low-confidence outputs
LOW_CONFIDENCE_THRESHOLD = 0.40


class AudioPredictor:
    """
    Singleton production inference engine for infant cry classification.

    Thread-safe via double-checked locking.
    Designed to be initialized once at server startup (Flask/FastAPI).

    Usage:
        predictor = AudioPredictor.get_instance(model_dir="./models")
        result = predictor.predict("path/to/cry.wav")
        print(result["detected_condition"])  # e.g., "hungry"
    """

    _instance: Optional["AudioPredictor"] = None
    _lock = threading.Lock()

    def __init__(self, model_dir: str):
        """
        Do not call directly. Use AudioPredictor.get_instance() instead.
        """
        self._model_dir = model_dir
        self._rf_model = None
        self._xgb_model = None
        self._scaler = None
        self._label_encoder = None
        self._feature_names = None
        self._training_meta = None
        self._extractor = FeatureExtractor()
        self._loaded = False
        self._load_models()

    @classmethod
    def get_instance(cls, model_dir: str = "models") -> "AudioPredictor":
        """
        Thread-safe singleton factory.

        Parameters:
            model_dir : str — path to directory containing trained artifacts

        Returns:
            AudioPredictor — shared singleton instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    logger.info(
                        f"[AudioPredictor] Creating singleton instance from: {model_dir}"
                    )
                    cls._instance = cls(model_dir)
        return cls._instance

    def _load_models(self) -> None:
        """
        Load all trained artifacts from disk.

        Raises:
            FileNotFoundError: if any required artifact is missing.
            RuntimeError     : if any artifact fails to unpickle.
        """
        required_files = {
            "rf_model": RF_MODEL_FILE,
            "xgb_model": XGB_MODEL_FILE,
            "scaler": SCALER_FILE,
            "label_encoder": LABEL_ENCODER_FILE,
        }

        model_dir = Path(self._model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(
                f"[AudioPredictor] Model directory not found: {self._model_dir}"
            )

        loaded = {}
        for key, filename in required_files.items():
            path = model_dir / filename
            if not path.exists():
                raise FileNotFoundError(
                    f"[AudioPredictor] Required artifact missing: {path}\n"
                    f"Have you run train_audio_model.py first?"
                )
            try:
                with open(path, "rb") as f:
                    loaded[key] = pickle.load(f)
                logger.info(f"[AudioPredictor] Loaded: {filename}")
            except Exception as e:
                raise RuntimeError(
                    f"[AudioPredictor] Failed to load '{filename}': {e}"
                ) from e

        self._rf_model = loaded["rf_model"]
        self._xgb_model = loaded["xgb_model"]
        self._scaler = loaded["scaler"]
        self._label_encoder = loaded["label_encoder"]

        # Optional artifacts (non-fatal if missing)
        for attr, filename in [
            ("_feature_names", FEATURE_NAMES_FILE),
            ("_training_meta", TRAINING_META_FILE),
        ]:
            path = model_dir / filename
            if path.exists():
                with open(path, "rb") as f:
                    setattr(self, attr, pickle.load(f))

        self._loaded = True
        classes = list(self._label_encoder.classes_)
        logger.info(
            f"[AudioPredictor] ✓ All models loaded. "
            f"Classes: {classes} | Feature dim: {self._extractor.feature_dim}"
        )

    def _validate_feature_vector(self, features: np.ndarray) -> None:
        """Internal check on extracted feature vector."""
        if np.any(np.isnan(features)):
            raise ValueError(
                "[AudioPredictor] Feature vector contains NaN values. "
                "Audio may be silent, corrupt, or too short."
            )
        if np.any(np.isinf(features)):
            raise ValueError(
                "[AudioPredictor] Feature vector contains Inf values."
            )

    def _run_ensemble(self, features_scaled: np.ndarray) -> Dict[str, float]:
        """
        Run ensemble inference and return per-class probability dict.

        Ensemble: 0.7 * RF_proba + 0.3 * XGB_proba

        Parameters:
            features_scaled: np.ndarray of shape (1, feature_dim)

        Returns:
            dict mapping class_name → probability (sums to 1.0)
        """
        rf_proba = self._rf_model.predict_proba(features_scaled)[0]
        xgb_proba = self._xgb_model.predict_proba(features_scaled)[0]

        # Weighted soft-probability fusion
        ensemble_proba = RF_WEIGHT * rf_proba + XGB_WEIGHT * xgb_proba

        # Normalize (should already sum to 1 but ensure numerical stability)
        ensemble_proba /= ensemble_proba.sum()

        classes = self._label_encoder.classes_
        proba_dict = {
            cls: round(float(p), 6)
            for cls, p in zip(classes, ensemble_proba)
        }
        return proba_dict

    def predict_from_array(
        self,
        y: np.ndarray,
        sr: int,
        audio_meta: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Run inference on a raw audio array.

        Parameters:
            y          : np.ndarray — mono float32 waveform
            sr         : int        — sample rate
            audio_meta : dict       — optional metadata to include in output

        Returns:
            dict — structured prediction output (JSON-serializable)
        """
        if not self._loaded:
            raise RuntimeError(
                "[AudioPredictor] Models not loaded. Call get_instance() first."
            )

        # Extract features
        try:
            features = self._extractor.extract(y, sr)
        except Exception as e:
            raise RuntimeError(
                f"[AudioPredictor] Feature extraction failed: {e}"
            ) from e

        self._validate_feature_vector(features)

        # Scale features using the SAME scaler fitted during training
        features_scaled = self._scaler.transform(features.reshape(1, -1))

        # Ensemble inference
        proba_dict = self._run_ensemble(features_scaled)

        # Sort by probability descending
        sorted_probs = sorted(proba_dict.items(), key=lambda x: x[1], reverse=True)

        detected_condition = sorted_probs[0][0]
        confidence = sorted_probs[0][1]
        secondary_condition = sorted_probs[1][0] if len(sorted_probs) > 1 else None

        # Flag low-confidence predictions
        is_low_confidence = confidence < LOW_CONFIDENCE_THRESHOLD
        if is_low_confidence:
            logger.warning(
                f"[AudioPredictor] Low confidence prediction: "
                f"{detected_condition} ({confidence:.3f}). "
                f"Probability mass is spread across classes."
            )

        result = {
            "module": "audio",
            "detected_condition": detected_condition,
            "confidence": round(confidence, 4),
            "secondary_condition": secondary_condition,
            "all_probabilities": {k: round(v, 4) for k, v in sorted(proba_dict.items())},
            "low_confidence_warning": is_low_confidence,
            "meta": {
                "model": "ensemble_rf_xgb",
                "ensemble_weights": {"rf": RF_WEIGHT, "xgb": XGB_WEIGHT},
                "feature_dim": int(self._extractor.feature_dim),
                "audio_duration_seconds": round(len(y) / sr, 3),
                "audio_warnings": (audio_meta or {}).get("warnings", []),
            },
        }

        logger.info(
            f"[AudioPredictor] Prediction: '{detected_condition}' "
            f"(confidence: {confidence:.4f}) | "
            f"2nd: '{secondary_condition}'"
        )
        return result

    def predict(self, filepath: str) -> Dict[str, Any]:
        """
        Run inference on an audio file (.wav or .mp3).

        Parameters:
            filepath : str — path to audio file

        Returns:
            dict — structured prediction output (JSON-serializable)

        Example output:
            {
                "module": "audio",
                "detected_condition": "hungry",
                "confidence": 0.84,
                "secondary_condition": "discomfort",
                "all_probabilities": {
                    "belly_pain": 0.04,
                    "burping": 0.0,
                    "discomfort": 0.09,
                    "hungry": 0.84,
                    "tired": 0.03
                },
                "low_confidence_warning": false,
                "meta": { ... }
            }
        """
        logger.info(f"[AudioPredictor] Predicting: {filepath}")

        # Load with quality checks
        y, sr, audio_meta = load_audio_with_checks(filepath)

        return self.predict_from_array(y, sr, audio_meta=audio_meta)

    def predict_batch(self, filepaths: list) -> list:
        """
        Run inference on a list of audio files.

        Parameters:
            filepaths : list of str — list of audio file paths

        Returns:
            list of dicts — one prediction result per file
        """
        results = []
        for i, filepath in enumerate(filepaths):
            try:
                result = self.predict(filepath)
                result["filepath"] = filepath
                result["index"] = i
                results.append(result)
            except Exception as e:
                logger.error(
                    f"[AudioPredictor] Failed on file {i} '{filepath}': {e}"
                )
                results.append({
                    "module": "audio",
                    "filepath": filepath,
                    "index": i,
                    "error": str(e),
                    "detected_condition": None,
                    "confidence": None,
                })
        return results

    @property
    def classes(self) -> list:
        """Return list of class names the model was trained on."""
        return list(self._label_encoder.classes_)

    @property
    def feature_dim(self) -> int:
        """Return expected feature vector dimensionality."""
        return self._extractor.feature_dim

    @property
    def is_loaded(self) -> bool:
        """Return True if models are loaded and ready."""
        return self._loaded


# ─── FLASK INTEGRATION EXAMPLE ────────────────────────────────────────────────
#
#   from flask import Flask, request, jsonify
#   from audio_module.audio_predictor import AudioPredictor
#
#   app = Flask(__name__)
#
#   @app.before_first_request
#   def load_models():
#       AudioPredictor.get_instance(model_dir="./models")
#
#   @app.route("/predict", methods=["POST"])
#   def predict():
#       file = request.files["audio"]
#       filepath = f"/tmp/{file.filename}"
#       file.save(filepath)
#       predictor = AudioPredictor.get_instance()
#       result = predictor.predict(filepath)
#       return jsonify(result)
#
# ─────────────────────────────────────────────────────────────────────────────