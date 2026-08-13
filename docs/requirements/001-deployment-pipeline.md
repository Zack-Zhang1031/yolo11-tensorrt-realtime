# YOLO11 Deployment Pipeline Requirements

## Overview

Provide a complete YOLO11s PyTorch-to-ONNX-to-TensorRT FP16 deployment codebase with real-time
inference entry points and synchronized performance measurement.

## User Stories

- As a CV engineer, I can run image, video, and webcam inference through a stable result contract.
- As a deployment engineer, I can export ONNX and build/run TensorRT when optional dependencies exist.
- As a reviewer, I can run synchronized benchmark measurements under documented conditions.

## Functional Reqs

- Load official or user-supplied YOLO11s weights.
- Return class ID, name, confidence, and XYXY box dictionaries.
- Export and validate ONNX; infer with provider selection, NMS, and coordinate projection.
- Build FP16 TensorRT 10.x engines and manage CUDA inference buffers/streams.
- Benchmark PyTorch, ONNX Runtime, and TensorRT using shared statistics.

## Non-Functional Reqs

- Support Python 3.10+, Windows and Linux paths, typed modular code, and clean exceptions.
- Keep ONNX Runtime and TensorRT behavior observable; TensorRT remains optional.
- Do not train, download datasets, commit artifacts, or invent performance/accuracy results.

## Data Model

Each detection contains `class_id`, `class_name`, `confidence`, and `bbox: [x1, y1, x2, y2]`.
Benchmark results contain backend, precision, size, batch, warmup/runs, mean, median, P95, and FPS.

## UI/UX

Command-line scripts provide actionable errors. Video and camera windows show labels and FPS and
exit with Esc/Q. An unavailable camera exits promptly.

## API

`YOLODetector.predict`, `ONNXDetector.predict`, and `TensorRTDetector.predict` return the common
detection contract. `run_benchmark` accepts inference and synchronization callables.

## Testing

Run compileall, pytest, smoke test, one real YOLO11s inference, short PyTorch benchmark, and ONNX /
TensorRT validation when dependencies are available.

## Open Questions

- Which CUDA/TensorRT production matrix will be the long-term supported target?
- Should later releases support dynamic image shapes or end-to-end NMS exports?
