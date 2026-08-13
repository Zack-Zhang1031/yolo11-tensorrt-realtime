# Inference Backends

## Overview

The package exposes PyTorch/Ultralytics, ONNX Runtime, and TensorRT implementations behind the same
plain-dictionary output contract.

## Design Decisions

- Keep Ultralytics `Results` inside `YOLODetector`.
- Share letterbox and raw detection decoding between ONNX Runtime and TensorRT.
- Discover ONNX tensor shape and metadata instead of hard-coding anchor counts.
- Make explicit ONNX CUDA selection strict while keeping `auto` fallback behavior.
- Use TensorRT 10.x named tensor APIs and lazy optional imports.
- Keep the TensorRT builder in the importable package and the CLI as a thin adapter.
- Use pinned host buffers and rebuild allocations when a dynamic input shape changes.
- Share one typed capture loop between video and camera applications.
- Pass an explicit synchronization callback into benchmark timing.

## Implementation Notes

ONNX Runtime `auto` mode prefers CUDA and permits CPU fallback. Explicit `cuda` mode validates the
active provider and disables runtime fallback. TensorRT buffers are reused for the selected input
shape and released by `close()` or the context manager. Cleanup attempts every host buffer, device
buffer, and stream even if an earlier release fails. Raw outputs are decoded as YOLO11 detection
tensors with four box coordinates followed by class probabilities.
