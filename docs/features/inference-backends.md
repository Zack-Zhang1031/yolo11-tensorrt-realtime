# Inference Backends

## Overview

The package exposes PyTorch/Ultralytics, ONNX Runtime, and TensorRT implementations behind the same
plain-dictionary output contract.

## Design Decisions

- Keep Ultralytics `Results` inside `YOLODetector`.
- Share letterbox and raw detection decoding between ONNX Runtime and TensorRT.
- Discover ONNX tensor shape and metadata instead of hard-coding anchor counts.
- Use TensorRT 10.x named tensor APIs and lazy optional imports.
- Pass an explicit synchronization callback into benchmark timing.

## Implementation Notes

ONNX Runtime selects CPU by default and CUDA when the provider exists. TensorRT buffers are reused
for the lifetime of the context and released by `close()` or the context manager. Raw outputs are
decoded as YOLO11 detection tensors with four box coordinates followed by class probabilities.

