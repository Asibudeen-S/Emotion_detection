# ExpressVision AI

An interactive Streamlit dashboard for multi-face emotion and facial attribute analysis. It supports uploaded JPG/PNG images and browser camera video through WebRTC. DeepFace performs face detection, apparent age and gender estimates, emotion scoring, and optional face embedding comparisons; OpenCV optical flow maintains track IDs and moves annotations between analysis frames.

## Features

- Detect all faces in an uploaded image and draw emotion-colored boxes.
- Show the seven emotion confidence scores for each face.
- Report estimated age, gender, dominant emotion, detector confidence, and box coordinates.
- Optionally compare detected faces with the largest face in a reference image using FaceNet512 embeddings and cosine similarity.
- Process live camera video in the browser, track faces between DeepFace analysis passes, and report FPS and frame/analysis latency.
- Keep uploaded images and embeddings in application memory; this code does not save them to disk.

## Run locally

Use Python 3.10, 3.11, or 3.12. DeepFace installs TensorFlow and may download its model weights on the first analysis, so allow time and disk space for initial setup.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

If TensorFlow reports that `tf-keras` is missing, make sure the active environment has the matching version installed with `python -m pip install -r requirements.txt`, then restart Streamlit using `python -m streamlit run app.py`.

Open the local Streamlit URL in a modern browser. Allow camera access when prompted. The camera transport uses WebRTC; remote deployments need HTTPS and a suitable STUN/TURN configuration for their network. The bundled public STUN address is only a basic connectivity default; replace it with infrastructure appropriate to your deployment.

## Run in Docker

```bash
docker build -t expressvision-ai .
docker run --rm -p 8501:8501 -v expressvision-models:/home/app/.deepface expressvision-ai
```

Put the container behind an HTTPS reverse proxy for remote camera access. The named volume keeps DeepFace model weights between container restarts.

## Use

1. Open **Image analysis**, upload a JPG/PNG, and select **Analyze image**.
2. Use the face selector to inspect each face's confidence distribution.
3. Optionally upload a reference photo in the sidebar. When more than one face is detected, the largest is used as the reference.
4. Open **Live camera** and start the camera. The model analyzes at the configured interval while OpenCV tracking and annotations run on each received frame.
5. Set the cosine similarity threshold in the sidebar. A higher threshold is more selective; calibrate it with representative, consented data before relying on a match decision.

## Notes for deployment

- Run behind an HTTPS reverse proxy and configure WebRTC network access (STUN/TURN) for your users.
- Place an authentication and rate-limiting layer in front of Streamlit before exposing it beyond a trusted network. This starter does not implement account management.
- DeepFace inference is CPU/GPU intensive. Concurrency is serialized within a process to protect model calls; use a controlled worker count and size server resources for the expected number of active streams.
- The default OpenCV detector favors availability and speed. Faces that are small, occluded, rotated, or poorly lit may be missed. Replace the detector backend and add the corresponding dependency if the deployment needs a different accuracy/performance tradeoff.
- Apparent age, gender, and emotion are uncertain model estimates. They do not establish a person's identity, internal emotional state, age, or gender. Do not use them as the sole basis for consequential decisions.
- Face embeddings are biometric data. Obtain appropriate consent, define retention rules, and do not treat the default similarity threshold as universally calibrated.

## Project layout

```text
app.py                    Streamlit dashboard and WebRTC processor
expressvision/analysis.py DeepFace output normalization and embeddings
expressvision/tracking.py OpenCV optical-flow face tracker
expressvision/rendering.py Emotion-colored frame annotation
requirements.txt          Python dependencies
```

## DeepFace API references

- [DeepFace analyze and represent APIs](https://github.com/serengil/deepface/blob/master/deepface/DeepFace.py)
- [DeepFace representation module](https://github.com/serengil/deepface/blob/master/deepface/modules/representation.py)
- [streamlit-webrtc project and examples](https://github.com/whitphx/streamlit-webrtc)
