# Project Progress

## Current Capabilities

- YOLO11s PyTorch inference adapter
- Image, video, and webcam command-line applications
- Static and dynamic ONNX export
- ONNX Runtime CPU/CUDA provider selection
- TensorRT 10.x FP16 engine generation and inference runtime
- Synchronized backend benchmarks
- Automated unit and smoke tests

## Engineering Guardrails

- Generated model artifacts and datasets remain outside Git.
- Metrics are published only with their measurement conditions.
- TensorRT dependencies remain isolated from the core inference package.
- Model architecture changes stay outside the deployment layer.
