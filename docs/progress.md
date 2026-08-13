# Project Progress

## Current Capabilities

- YOLO11s PyTorch inference adapter
- Image, video, and webcam command-line applications
- Static and dynamic ONNX export
- ONNX graph checking and strict/automatic provider modes
- ONNX Runtime CPU/CUDA provider selection
- TensorRT 10.x FP16 engine generation, dynamic shapes, and pinned-memory inference
- Synchronized backend benchmarks
- Python 3.10-3.13 CI, static analysis, unit tests, and hardware integration test entry point

## Engineering Guardrails

- Generated model artifacts and datasets remain outside Git.
- Metrics are published only with their measurement conditions.
- TensorRT dependencies remain isolated from the core inference package.
- Model architecture changes stay outside the deployment layer.
