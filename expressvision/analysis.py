"""DeepFace-backed facial attribute and embedding analysis."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any, Mapping

import cv2
import numpy as np


EMOTION_KEYS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")
EMOTION_LABELS = {
    "angry": "Angry",
    "disgust": "Disgust",
    "fear": "Fear",
    "happy": "Happy",
    "sad": "Sad",
    "surprise": "Surprise",
    "neutral": "Neutral",
}


@dataclass(frozen=True, slots=True)
class FaceObservation:
    """A single detected face and its model estimates."""

    bbox: tuple[int, int, int, int]
    detector_confidence: float
    age: int | None
    gender: str
    dominant_emotion: str
    emotions: Mapping[str, float]
    embedding_similarity: float | None = None
    embedding_verified: bool | None = None


class DeepFaceAnalyzer:
    """Serialize model calls and normalize DeepFace output for the UI."""

    def __init__(self, detector_backend: str = "opencv") -> None:
        self.detector_backend = detector_backend
        self._lock = RLock()
        self._deepface: Any | None = None

    def _api(self) -> Any:
        if self._deepface is None:
            try:
                from deepface import DeepFace
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise RuntimeError(
                    "DeepFace is not installed. Install the project requirements first."
                ) from exc
            self._deepface = DeepFace
        return self._deepface

    def _analyze_raw(self, image_bgr: np.ndarray) -> list[dict[str, Any]]:
        if image_bgr is None or image_bgr.size == 0:
            return []
        result = self._api().analyze(
            img_path=image_bgr,
            actions=["age", "gender", "emotion"],
            detector_backend=self.detector_backend,
            enforce_detection=False,
            align=True,
            silent=True,
        )
        if isinstance(result, dict):
            return [result]
        if not isinstance(result, (list, tuple)):
            return []
        # DeepFace returns a list of face dictionaries for a single image. Handle
        # nested results too, so this remains safe if batch output is introduced.
        return [item for group in result for item in (group if isinstance(group, list) else [group]) if isinstance(item, dict)]

    @staticmethod
    def _bbox(region: Mapping[str, Any], width: int, height: int) -> tuple[int, int, int, int]:
        x = max(0, min(width - 1, int(region.get("x", 0))))
        y = max(0, min(height - 1, int(region.get("y", 0))))
        w = max(0, min(width - x, int(region.get("w", 0))))
        h = max(0, min(height - y, int(region.get("h", 0))))
        return x, y, w, h

    @staticmethod
    def _normalize_emotions(value: Any) -> dict[str, float]:
        source = value if isinstance(value, Mapping) else {}
        normalized = {key: 0.0 for key in EMOTION_KEYS}
        for key, score in source.items():
            emotion_key = str(key).strip().lower()
            if emotion_key in normalized:
                try:
                    normalized[emotion_key] = max(0.0, float(score))
                except (TypeError, ValueError):
                    continue
        total = sum(normalized.values())
        if total > 0 and total < 1.01:
            normalized = {key: score * 100.0 for key, score in normalized.items()}
        return normalized

    @staticmethod
    def _gender(face: Mapping[str, Any]) -> str:
        dominant = face.get("dominant_gender")
        if dominant:
            return str(dominant).title()
        gender_scores = face.get("gender")
        if isinstance(gender_scores, Mapping) and gender_scores:
            return str(max(gender_scores, key=gender_scores.get)).title()
        return "Unknown"

    @staticmethod
    def _face_crop(image_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
        x, y, width, height = bbox
        if width < 16 or height < 16:
            return None
        crop = image_bgr[y : y + height, x : x + width]
        return crop if crop.size else None

    def _embedding(self, face_crop_bgr: np.ndarray) -> np.ndarray | None:
        # The crop is already a detected face; skip runs the recognition model on
        # the full crop and avoids a second detector pass.
        result = self._api().represent(
            img_path=face_crop_bgr,
            model_name="Facenet512",
            detector_backend="skip",
            enforce_detection=False,
            align=False,
        )
        if not result:
            return None
        vector = np.asarray(result[0].get("embedding", []), dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if vector.ndim != 1 or vector.size == 0 or norm == 0 or not np.isfinite(vector).all():
            return None
        return vector / norm

    @staticmethod
    def _similarity(first: np.ndarray, second: np.ndarray) -> float | None:
        if first.shape != second.shape or first.size == 0:
            return None
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        if denominator == 0:
            return None
        return float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))

    def create_reference_embedding(self, image_bgr: np.ndarray) -> tuple[np.ndarray, int]:
        """Embed the largest visible face in a reference image."""
        with self._lock:
            raw_faces = self._analyze_raw(image_bgr)
            height, width = image_bgr.shape[:2]
            valid: list[tuple[int, dict[str, Any], tuple[int, int, int, int]]] = []
            for face in raw_faces:
                region = face.get("region", {})
                bbox = self._bbox(region if isinstance(region, Mapping) else {}, width, height)
                x, y, box_width, box_height = bbox
                confidence_value = face.get("face_confidence", region.get("confidence", 0.0)) if isinstance(region, Mapping) else 0.0
                confidence = float(confidence_value or 0.0)
                if box_width >= 16 and box_height >= 16 and confidence > 0:
                    valid.append((box_width * box_height, face, bbox))
            if not valid:
                raise ValueError("No face was detected in the reference image.")
            _, _, bbox = max(valid, key=lambda item: item[0])
            crop = self._face_crop(image_bgr, bbox)
            if crop is None:
                raise ValueError("The detected reference face is too small to embed.")
            vector = self._embedding(crop)
            if vector is None:
                raise ValueError("DeepFace could not create an embedding for the reference face.")
            return vector, len(valid)

    def analyze_faces(
        self,
        image_bgr: np.ndarray,
        reference_embedding: np.ndarray | None = None,
        verification_threshold: float = 0.70,
    ) -> list[FaceObservation]:
        """Analyze all detected faces in a BGR frame or image."""
        if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("Expected a three-channel BGR image.")
        with self._lock:
            raw_faces = self._analyze_raw(image_bgr)
            frame_height, frame_width = image_bgr.shape[:2]
            observations: list[FaceObservation] = []
            for face in raw_faces:
                region_value = face.get("region", {})
                region = region_value if isinstance(region_value, Mapping) else {}
                bbox = self._bbox(region, frame_width, frame_height)
                x, y, width, height = bbox
                confidence_value = face.get("face_confidence", region.get("confidence", 0.0))
                try:
                    confidence = float(confidence_value or 0.0)
                except (TypeError, ValueError):
                    confidence = 0.0
                # With enforce_detection=False DeepFace can return a whole-image
                # fallback region. Exclude those entries from face analytics.
                if width < 16 or height < 16 or confidence <= 0:
                    continue

                emotions = self._normalize_emotions(face.get("emotion"))
                dominant_key = str(face.get("dominant_emotion") or max(emotions, key=emotions.get)).lower()
                if dominant_key not in EMOTION_LABELS:
                    dominant_key = max(emotions, key=emotions.get)
                age_value = face.get("age")
                try:
                    age = int(round(float(age_value))) if age_value is not None else None
                except (TypeError, ValueError):
                    age = None

                similarity: float | None = None
                verified: bool | None = None
                if reference_embedding is not None:
                    crop = self._face_crop(image_bgr, bbox)
                    candidate_embedding = self._embedding(crop) if crop is not None else None
                    if candidate_embedding is not None:
                        similarity = self._similarity(reference_embedding, candidate_embedding)
                        if similarity is not None:
                            verified = similarity >= verification_threshold

                observations.append(
                    FaceObservation(
                        bbox=bbox,
                        detector_confidence=confidence,
                        age=age,
                        gender=self._gender(face),
                        dominant_emotion=EMOTION_LABELS[dominant_key],
                        emotions=emotions,
                        embedding_similarity=similarity,
                        embedding_verified=verified,
                    )
                )
            return observations
