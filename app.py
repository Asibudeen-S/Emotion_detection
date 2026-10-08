"""ExpressVision AI — interactive facial attribute and emotion dashboard."""

from __future__ import annotations

from collections import deque
from hashlib import sha256
from io import BytesIO
from threading import RLock
import time
from typing import Any

import av
import cv2
import logging
import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

from expressvision.analysis import (
    EMOTION_KEYS,
    EMOTION_LABELS,
    DeepFaceAnalyzer,
    FaceObservation,
)
from expressvision.rendering import annotate_image, annotate_tracked_frame
from expressvision.tracking import FaceTracker


LOGGER = logging.getLogger("expressvision")

st.set_page_config(
    page_title="ExpressVision AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { --ink: #f4f6fa; --muted: #96a2b6; --surface: #111925; --border: #243147; --lime: #c9f36b; }
    html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
    .stApp { background: radial-gradient(ellipse at 12% -10%, #1b2b3a 0, #0a1018 48%, #090e15 100%); color: var(--ink); }
    [data-testid="stSidebar"] { background: #0c131d; border-right: 1px solid var(--border); }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { letter-spacing: -.03em; }
    .hero { padding: 1.2rem 0 1.5rem; border-bottom: 1px solid var(--border); margin-bottom: 1.5rem; }
    .eyebrow { color: var(--lime); font: 500 .72rem 'DM Mono', monospace; letter-spacing: .18em; text-transform: uppercase; }
    .hero h1 { font-size: clamp(2rem, 4vw, 3.15rem); letter-spacing: -.065em; line-height: 1.05; margin: .45rem 0 .55rem; }
    .hero p { color: var(--muted); font-size: 1rem; margin: 0; max-width: 780px; }
    .section-kicker { color: var(--lime); font: 500 .7rem 'DM Mono', monospace; letter-spacing: .15em; text-transform: uppercase; margin-bottom: .25rem; }
    .metric-label { color: var(--muted); font: 500 .68rem 'DM Mono', monospace; letter-spacing: .11em; text-transform: uppercase; }
    .soft-note { color: var(--muted); font-size: .82rem; }
    div[data-testid="stMetric"] { background: rgba(17,25,37,.86); border: 1px solid var(--border); padding: .95rem 1rem; border-radius: 12px; }
    div[data-testid="stMetricLabel"] { color: var(--muted); font: 500 .68rem 'DM Mono', monospace; letter-spacing: .08em; text-transform: uppercase; }
    .stTabs [data-baseweb="tab-list"] { gap: .45rem; border-bottom: 1px solid var(--border); }
    .stTabs [data-baseweb="tab"] { color: var(--muted); height: 3rem; }
    .stTabs [aria-selected="true"] { color: var(--lime) !important; }
    .stButton > button[kind="primary"] { background: var(--lime); color: #111820; border: 0; font-weight: 800; }
    .stButton > button[kind="primary"]:hover { background: #d9ff8e; color: #111820; }
    div[data-testid="stFileUploader"] section { border-color: #34445b; background: rgba(17,25,37,.65); }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_analyzer() -> DeepFaceAnalyzer:
    return DeepFaceAnalyzer(detector_backend="opencv")


def image_bytes_to_bgr(raw_bytes: bytes) -> np.ndarray:
    with Image.open(BytesIO(raw_bytes)) as image:
        rgb = np.asarray(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def emotion_chart(emotions: dict[str, float], title: str = "Emotion confidence") -> go.Figure:
    labels = [EMOTION_LABELS[key] for key in EMOTION_KEYS]
    values = [float(emotions.get(key, 0.0)) for key in EMOTION_KEYS]
    palette = [
        "#f06670", "#80c889", "#c28be0", "#d3f279", "#70a9e1", "#f5a04d", "#8b9aaf"
    ]
    figure = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=palette,
            marker_line_width=0,
            text=[f"{value:.0f}%" for value in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
        )
    )
    figure.update_layout(
        title={"text": title, "font": {"size": 15, "color": "#f4f6fa"}},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#aab5c6", "family": "Manrope, sans-serif", "size": 11},
        margin={"l": 12, "r": 12, "t": 46, "b": 8},
        height=310,
        yaxis={"range": [0, max(100, max(values, default=0) * 1.12)], "ticksuffix": "%", "gridcolor": "#263246", "zeroline": False},
        xaxis={"showgrid": False, "fixedrange": True},
        showlegend=False,
    )
    return figure


def face_frame(face: FaceObservation, index: int) -> dict[str, Any]:
    identity = "Reference not set"
    if face.embedding_verified is not None:
        identity = "Verified" if face.embedding_verified else "No match"
    similarity = "—" if face.embedding_similarity is None else f"{face.embedding_similarity:.3f}"
    x, y, width, height = face.bbox
    return {
        "Face": f"Face {index + 1}",
        "Age*": face.age if face.age is not None else "—",
        "Gender estimate*": face.gender,
        "Dominant emotion": face.dominant_emotion,
        "Face confidence": f"{face.detector_confidence * 100:.0f}%",
        "Embedding check": identity,
        "Cosine similarity": similarity,
        "Box (x, y, w, h)": f"{x}, {y}, {width}, {height}",
    }


def render_face_analytics(faces: list[FaceObservation], key_prefix: str) -> None:
    if not faces:
        st.info("No faces were detected. Try a well-lit image with clearly visible faces.")
        return
    selected_label = st.selectbox(
        "Face confidence profile",
        options=[f"Face {index + 1} · {face.dominant_emotion}" for index, face in enumerate(faces)],
        key=f"{key_prefix}-face-select",
    )
    selected_index = next(
        index for index, face in enumerate(faces) if selected_label.startswith(f"Face {index + 1} ·")
    )
    selected = faces[selected_index]
    chart_col, details_col = st.columns([1.15, 1], gap="large")
    with chart_col:
        st.plotly_chart(
            emotion_chart(selected.emotions, f"Face {selected_index + 1} · emotion distribution"),
            use_container_width=True,
            config={"displayModeBar": False},
            key=f"{key_prefix}-emotion-chart",
        )
    with details_col:
        st.markdown('<div class="section-kicker">Face attributes</div>', unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame([face_frame(face, index) for index, face in enumerate(faces)]),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("* Apparent age and gender are model estimates. Embedding similarity is a configurable biometric comparison score.")


class ExpressVisionVideoProcessor(VideoProcessorBase):
    """Per-browser WebRTC processor with detection cadence, tracking and metrics."""

    def __init__(
        self,
        analyzer: DeepFaceAnalyzer,
        reference_embedding: np.ndarray | None,
        verification_threshold: float,
        inference_interval: float,
    ) -> None:
        self.analyzer = analyzer
        self.reference_embedding = reference_embedding
        self.verification_threshold = verification_threshold
        self.inference_interval = inference_interval
        self.tracker = FaceTracker(max_track_age_seconds=max(2.5, inference_interval * 3.0))
        self._next_inference_at = 0.0
        self._metrics_lock = RLock()
        self._frame_times: deque[float] = deque(maxlen=75)
        self._latency_history: deque[float] = deque(maxlen=45)
        self._last_latency_ms = 0.0
        self._last_inference_ms: float | None = None
        self._latest_error: str | None = None
        self._latest_faces: list[dict[str, Any]] = []
        self._last_frame_at: float | None = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        started = time.perf_counter()
        now = time.monotonic()
        image_bgr = frame.to_ndarray(format="bgr24")
        should_infer = now >= self._next_inference_at
        if should_infer:
            inference_started = time.perf_counter()
            analysis_error = None
            try:
                detections = self.analyzer.analyze_faces(
                    image_bgr,
                    self.reference_embedding,
                    self.verification_threshold,
                )
                self.tracker.update(image_bgr, detections, time.monotonic())
            except Exception as exc:
                LOGGER.exception("Live DeepFace inference failed")
                analysis_error = "Analysis failed; see the server logs for details."
                self.tracker.update(image_bgr, None, time.monotonic())
            inference_latency_ms = (time.perf_counter() - inference_started) * 1000.0
            self._next_inference_at = time.monotonic() + self.inference_interval
        else:
            analysis_error = None
            inference_latency_ms = None
            self.tracker.update(image_bgr, None, now)

        annotated = annotate_tracked_frame(image_bgr, self.tracker.tracks)
        output = av.VideoFrame.from_ndarray(annotated, format="bgr24")
        finished = time.perf_counter()
        elapsed_ms = (finished - started) * 1000.0
        with self._metrics_lock:
            self._frame_times.append(finished)
            self._latency_history.append(elapsed_ms)
            self._last_latency_ms = elapsed_ms
            if should_infer:
                self._latest_error = analysis_error
                self._last_inference_ms = inference_latency_ms
            self._last_frame_at = now
            self._latest_faces = [
                {
                    "track_id": track.track_id,
                    "bbox": track.bbox,
                    "age": track.observation.age,
                    "gender": track.observation.gender,
                    "dominant_emotion": track.observation.dominant_emotion,
                    "emotions": dict(track.observation.emotions),
                    "embedding_similarity": track.observation.embedding_similarity,
                    "embedding_verified": track.observation.embedding_verified,
                }
                for track in self.tracker.tracks
            ]
        return output

    def get_metrics(self) -> dict[str, Any]:
        with self._metrics_lock:
            frame_times = list(self._frame_times)
            if len(frame_times) > 1 and frame_times[-1] > frame_times[0]:
                fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
            else:
                fps = 0.0
            return {
                "fps": fps,
                "frame_latency_ms": self._last_latency_ms,
                "inference_latency_ms": self._last_inference_ms,
                "face_count": len(self._latest_faces),
                "faces": [dict(face, emotions=dict(face["emotions"])) for face in self._latest_faces],
                "latency_history": list(self._latency_history),
                "error": self._latest_error,
                "last_frame_at": self._last_frame_at,
            }


@st.fragment(run_every=1.0)
def render_live_metrics(context: Any) -> None:
    processor = getattr(context, "video_processor", None)
    if processor is None:
        st.info("Start the camera to see live face analytics and performance metrics.")
        return
    metrics = processor.get_metrics()
    if metrics["error"]:
        st.warning(metrics["error"])

    fps_col, latency_col, inference_col, faces_col = st.columns(4)
    fps_col.metric("Video FPS", f"{metrics['fps']:.1f}")
    latency_col.metric("Frame latency", f"{metrics['frame_latency_ms']:.1f} ms")
    inference = metrics["inference_latency_ms"]
    inference_col.metric("Analysis latency", f"{inference:.0f} ms" if inference is not None else "—")
    faces_col.metric("Tracked faces", metrics["face_count"])

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        if metrics["latency_history"]:
            figure = go.Figure(
                go.Scatter(
                    y=metrics["latency_history"],
                    mode="lines",
                    line={"color": "#c9f36b", "width": 2},
                    fill="tozeroy",
                    fillcolor="rgba(201,243,107,.08)",
                    hovertemplate="Frame %{x}: %{y:.1f} ms<extra></extra>",
                )
            )
            figure.update_layout(
                title={"text": "Per-frame processing latency", "font": {"size": 15, "color": "#f4f6fa"}},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#aab5c6", "size": 11},
                margin={"l": 12, "r": 12, "t": 46, "b": 8},
                height=280,
                yaxis={"title": "milliseconds", "gridcolor": "#263246", "zeroline": False},
                xaxis={"title": "Recent frames", "showgrid": False},
                showlegend=False,
            )
            st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False}, key="live-latency-chart")
    with right:
        if metrics["faces"]:
            face_options = [f"Track #{face['track_id']} · {face['dominant_emotion']}" for face in metrics["faces"]]
            selected_label = st.selectbox("Live emotion profile", face_options, key="live-face-select")
            selected = metrics["faces"][face_options.index(selected_label)]
            st.plotly_chart(
                emotion_chart(selected["emotions"], f"Track #{selected['track_id']} · emotion distribution"),
                use_container_width=True,
                config={"displayModeBar": False},
                key="live-emotion-chart",
            )
            identity = "Reference not set"
            if selected["embedding_verified"] is not None:
                identity = "Verified" if selected["embedding_verified"] else "No match"
            similarity = selected["embedding_similarity"]
            st.caption(
                f"Age estimate: {selected['age'] if selected['age'] is not None else '—'} · "
                f"Gender estimate: {selected['gender']} · Embedding check: {identity}"
                + (f" · Cosine similarity: {similarity:.3f}" if similarity is not None else "")
            )
        else:
            st.markdown('<div class="section-kicker">Emotion confidence</div>', unsafe_allow_html=True)
            st.caption("No tracked faces yet. Keep a face inside the camera frame.")


analyzer = get_analyzer()

st.sidebar.markdown("## ExpressVision AI")
st.sidebar.markdown('<div class="section-kicker">Analysis controls</div>', unsafe_allow_html=True)
inference_interval = st.sidebar.slider(
    "Camera analysis interval",
    min_value=0.4,
    max_value=2.0,
    value=0.8,
    step=0.1,
    help="DeepFace runs at this interval; OpenCV tracking and annotation run on every video frame.",
)
verification_threshold = st.sidebar.slider(
    "Embedding match threshold",
    min_value=0.50,
    max_value=0.95,
    value=0.70,
    step=0.01,
    help="Cosine similarity at or above this value is marked as a match. Calibrate for your use case.",
)
st.sidebar.markdown('<div class="section-kicker" style="margin-top:1.3rem">Reference face · optional</div>', unsafe_allow_html=True)
reference_upload = st.sidebar.file_uploader(
    "Upload one image to enable identity matching",
    type=["jpg", "jpeg", "png"],
    key="reference-face-upload",
    help="If there are multiple faces, the largest detected face is used. The image and embedding remain in this session only.",
)

reference_embedding: np.ndarray | None = None
reference_signature: str | None = None
if reference_upload is not None:
    reference_bytes = reference_upload.getvalue()
    reference_signature = sha256(reference_bytes).hexdigest()
    saved_signature = st.session_state.get("reference_signature")
    if saved_signature != reference_signature:
        try:
            reference_bgr = image_bytes_to_bgr(reference_bytes)
            with st.sidebar.status("Preparing reference face…", expanded=False):
                vector, reference_faces = analyzer.create_reference_embedding(reference_bgr)
            st.session_state["reference_embedding"] = vector
            st.session_state["reference_signature"] = reference_signature
            st.session_state["reference_faces_detected"] = reference_faces
            st.session_state.pop("reference_error", None)
        except Exception as exc:
            st.session_state["reference_signature"] = reference_signature
            st.session_state["reference_embedding"] = None
            st.session_state["reference_error"] = str(exc)
    if st.session_state.get("reference_error"):
        st.sidebar.error(st.session_state["reference_error"])
    else:
        reference_embedding = st.session_state.get("reference_embedding")
        st.sidebar.success(
            f"Reference ready · {st.session_state.get('reference_faces_detected', 1)} detected face(s)"
        )
else:
    st.session_state.pop("reference_signature", None)
    st.session_state.pop("reference_embedding", None)
    st.session_state.pop("reference_error", None)
    st.session_state.pop("reference_faces_detected", None)

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Computer vision · live analysis</div>
      <h1>ExpressVision AI</h1>
      <p>Explore facial expression signals and visual attributes across still images and a live camera feed.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("Emotion, age, and gender are model estimates from visible facial signals; they may be inaccurate and should not be treated as facts about a person.")

image_tab, live_tab = st.tabs(["▧  Image analysis", "◉  Live camera"])

with image_tab:
    st.markdown('<div class="section-kicker">Upload stream</div>', unsafe_allow_html=True)
    uploaded_image = st.file_uploader(
        "Choose a JPG or PNG image",
        type=["jpg", "jpeg", "png"],
        key="analysis-image-upload",
        help="The dashboard detects and analyzes every face found in the selected image.",
    )
    current_image_signature: str | None = None
    current_image_bytes: bytes | None = None
    if uploaded_image is not None:
        current_image_bytes = uploaded_image.getvalue()
        current_image_signature = sha256(current_image_bytes).hexdigest()

    if current_image_bytes is not None:
        if st.button("Analyze image", type="primary", key="analyze-image-button"):
            try:
                image_bgr = image_bytes_to_bgr(current_image_bytes)
                with st.spinner("Detecting faces and estimating attributes…"):
                    start = time.perf_counter()
                    faces = analyzer.analyze_faces(image_bgr, reference_embedding, verification_threshold)
                    elapsed_ms = (time.perf_counter() - start) * 1000.0
                    annotated = annotate_image(image_bgr, faces)
                st.session_state["image_result"] = {
                    "key": (current_image_signature, reference_signature, verification_threshold),
                    "annotated": annotated,
                    "faces": faces,
                    "latency_ms": elapsed_ms,
                    "size": (image_bgr.shape[1], image_bgr.shape[0]),
                }
            except Exception as exc:
                st.error(f"Could not analyze the selected image: {exc}")

    result = st.session_state.get("image_result")
    expected_result_key = (current_image_signature, reference_signature, verification_threshold)
    if result and current_image_signature and result["key"] == expected_result_key:
        count = len(result["faces"])
        dominant_face = max(
            result["faces"],
            key=lambda face: face.emotions.get(face.dominant_emotion.lower(), 0.0),
            default=None,
        )
        dominant = dominant_face.dominant_emotion if dominant_face else "—"
        metric_a, metric_b, metric_c, metric_d = st.columns(4)
        metric_a.metric("Faces detected", count)
        metric_b.metric("Dominant signal", dominant)
        metric_c.metric("Processing time", f"{result['latency_ms']:.0f} ms")
        metric_d.metric("Image dimensions", f"{result['size'][0]} × {result['size'][1]}")
        image_col, guide_col = st.columns([1.15, 1], gap="large")
        with image_col:
            st.markdown('<div class="section-kicker">Annotated image</div>', unsafe_allow_html=True)
            annotated_rgb = cv2.cvtColor(result["annotated"], cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, use_container_width=True)
        with guide_col:
            render_face_analytics(result["faces"], "image")
        if not count:
            st.info("Upload another image or try a closer, better-lit view.")
    elif uploaded_image is None:
        st.markdown("Upload a photo to see face detection, attribute estimates, and the seven-emotion confidence distribution.")
    elif not result or result["key"][0] != current_image_signature:
        st.info("Select **Analyze image** to process this image.")
    else:
        st.info("The reference face or match threshold changed. Select **Analyze image** to refresh the comparison.")

with live_tab:
    st.markdown('<div class="section-kicker">WebRTC camera stream</div>', unsafe_allow_html=True)
    if reference_embedding is not None:
        st.caption("Reference matching is active for this camera session.")
    else:
        st.caption("Add a reference face in the sidebar to enable identity matching.")
    processor_factory = lambda: ExpressVisionVideoProcessor(
        analyzer=analyzer,
        reference_embedding=reference_embedding,
        verification_threshold=verification_threshold,
        inference_interval=inference_interval,
    )
    webrtc_key = f"expressvision-live-{reference_signature or 'no-reference'}-{verification_threshold:.2f}-{inference_interval:.1f}"
    context = webrtc_streamer(
        key=webrtc_key,
        video_processor_factory=processor_factory,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        async_processing=True,
    )
    render_live_metrics(context)

st.sidebar.markdown("---")
st.sidebar.caption("Images and embeddings are not written to disk by this app. Keep biometric comparisons consent-based and use an appropriate threshold for your environment.")
