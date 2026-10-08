"""Lightweight OpenCV optical-flow tracking between face detections."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Sequence

import cv2
import numpy as np

from .analysis import FaceObservation


@dataclass(slots=True)
class TrackedFace:
    track_id: int
    bbox: tuple[int, int, int, int]
    observation: FaceObservation
    points: np.ndarray | None
    last_detected_at: float


def intersection_over_union(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


class FaceTracker:
    """Associate detections with stable IDs and move boxes with Lucas-Kanade flow."""

    def __init__(self, max_track_age_seconds: float = 1.8, min_iou: float = 0.12) -> None:
        self.max_track_age_seconds = max_track_age_seconds
        self.min_iou = min_iou
        self._tracks: dict[int, TrackedFace] = {}
        self._next_id = 1
        self._previous_gray: np.ndarray | None = None

    @property
    def tracks(self) -> list[TrackedFace]:
        return list(self._tracks.values())

    @staticmethod
    def _features(gray: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
        x, y, width, height = bbox
        mask = np.zeros_like(gray)
        mask[y : y + height, x : x + width] = 255
        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=36,
            qualityLevel=0.01,
            minDistance=4,
            blockSize=5,
            mask=mask,
        )

    @staticmethod
    def _clamp_box(
        bbox: tuple[int, int, int, int], frame_width: int, frame_height: int
    ) -> tuple[int, int, int, int]:
        x, y, width, height = bbox
        x = max(0, min(frame_width - 1, x))
        y = max(0, min(frame_height - 1, y))
        width = max(0, min(frame_width - x, width))
        height = max(0, min(frame_height - y, height))
        return x, y, width, height

    def _propagate(self, gray: np.ndarray, frame_width: int, frame_height: int) -> None:
        if self._previous_gray is None:
            return
        for track in self._tracks.values():
            points = track.points
            if points is None or len(points) < 3:
                track.points = self._features(gray, track.bbox)
                continue
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self._previous_gray,
                gray,
                points,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
            )
            if next_points is None or status is None:
                track.points = self._features(gray, track.bbox)
                continue
            good_new = next_points[status.reshape(-1) == 1]
            good_old = points[status.reshape(-1) == 1]
            if len(good_new) >= 3:
                displacement = np.median(good_new - good_old, axis=0).reshape(-1)
                x, y, width, height = track.bbox
                moved = self._clamp_box(
                    (int(round(x + displacement[0])), int(round(y + displacement[1])), width, height),
                    frame_width,
                    frame_height,
                )
                if moved[2] >= 16 and moved[3] >= 16:
                    track.bbox = moved
                    track.points = good_new.reshape(-1, 1, 2).astype(np.float32)
                else:
                    track.points = None
            else:
                track.points = self._features(gray, track.bbox)

    def update(
        self,
        frame_bgr: np.ndarray,
        detections: Sequence[FaceObservation] | None = None,
        now: float | None = None,
    ) -> list[TrackedFace]:
        """Update tracks; pass detections only on scheduled detector frames."""
        if frame_bgr is None or frame_bgr.size == 0:
            return self.tracks
        now = time.monotonic() if now is None else now
        frame_height, frame_width = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        self._propagate(gray, frame_width, frame_height)
        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if now - track.last_detected_at <= self.max_track_age_seconds
        }

        if detections is not None:
            candidates: list[tuple[float, int, int]] = []
            for track_id, track in self._tracks.items():
                for detection_index, detection in enumerate(detections):
                    candidates.append((intersection_over_union(track.bbox, detection.bbox), track_id, detection_index))
            candidates.sort(reverse=True)
            matched_tracks: set[int] = set()
            matched_detections: set[int] = set()
            for overlap, track_id, detection_index in candidates:
                if overlap < self.min_iou:
                    break
                if track_id in matched_tracks or detection_index in matched_detections:
                    continue
                detection = detections[detection_index]
                track = self._tracks[track_id]
                track.bbox = self._clamp_box(detection.bbox, frame_width, frame_height)
                track.observation = detection
                track.points = self._features(gray, track.bbox)
                track.last_detected_at = now
                matched_tracks.add(track_id)
                matched_detections.add(detection_index)

            for detection_index, detection in enumerate(detections):
                if detection_index in matched_detections:
                    continue
                bbox = self._clamp_box(detection.bbox, frame_width, frame_height)
                track_id = self._next_id
                self._next_id += 1
                self._tracks[track_id] = TrackedFace(
                    track_id=track_id,
                    bbox=bbox,
                    observation=detection,
                    points=self._features(gray, bbox),
                    last_detected_at=now,
                )

        self._previous_gray = gray
        return self.tracks
