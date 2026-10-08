"""OpenCV frame annotation and small UI formatting helpers."""

from __future__ import annotations

import cv2
import numpy as np

from .analysis import FaceObservation
from .tracking import TrackedFace


EMOTION_COLORS_BGR = {
    "Angry": (64, 72, 238),
    "Disgust": (82, 170, 80),
    "Fear": (190, 112, 196),
    "Happy": (44, 195, 245),
    "Sad": (225, 140, 85),
    "Surprise": (30, 158, 255),
    "Neutral": (150, 160, 174),
}


def _label_for_observation(observation: FaceObservation, track_id: int | None = None) -> str:
    score = observation.emotions.get(observation.dominant_emotion.lower(), 0.0)
    prefix = f"#{track_id} " if track_id is not None else ""
    return f"{prefix}{observation.dominant_emotion} {score:.0f}%"


def _draw_face(
    image_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    observation: FaceObservation,
    track_id: int | None = None,
) -> None:
    x, y, width, height = bbox
    color = EMOTION_COLORS_BGR.get(observation.dominant_emotion, (190, 200, 210))
    cv2.rectangle(image_bgr, (x, y), (x + width, y + height), color, 2, cv2.LINE_AA)
    label = _label_for_observation(observation, track_id)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, thickness = 0.55, 1
    (label_width, label_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    top = max(0, y - label_height - baseline - 10)
    cv2.rectangle(
        image_bgr,
        (x, top),
        (x + label_width + 12, top + label_height + baseline + 8),
        color,
        -1,
    )
    cv2.putText(
        image_bgr,
        label,
        (x + 6, top + label_height + 2),
        font,
        font_scale,
        (18, 22, 28),
        thickness,
        cv2.LINE_AA,
    )


def annotate_image(image_bgr: np.ndarray, faces: list[FaceObservation]) -> np.ndarray:
    annotated = image_bgr.copy()
    for face in faces:
        _draw_face(annotated, face.bbox, face)
    return annotated


def annotate_tracked_frame(image_bgr: np.ndarray, tracks: list[TrackedFace]) -> np.ndarray:
    annotated = image_bgr.copy()
    for track in tracks:
        _draw_face(annotated, track.bbox, track.observation, track.track_id)
    return annotated
