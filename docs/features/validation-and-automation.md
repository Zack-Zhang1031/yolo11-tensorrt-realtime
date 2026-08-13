# Validation and Automation

## Overview

The validation stack covers source compilation, linting, static type analysis, unit tests, an
offline smoke check, ONNX graph checking, and an opt-in TensorRT hardware integration test.

## Design Decisions

- Run CPU-safe CI on Python 3.10 through 3.13.
- Keep TensorRT tests behind a marker and an explicit environment switch.
- Validate ONNX with the official checker before treating an export as an artifact.
- Keep exact development versions in `constraints.txt` and broad supported ranges in project
  metadata.
- Exercise CUDA cleanup paths with fakes so lifecycle regressions do not require a GPU.

## Implementation Notes

The standard gate is `compileall`, Ruff, Pyright, Pytest, and `smoke_test.py --offline`. Set
`YOLO_TRT_INTEGRATION=1` and `YOLO_TRT_ONNX` on a TensorRT 10.x NVIDIA runner to enable the engine
build-and-infer test.
