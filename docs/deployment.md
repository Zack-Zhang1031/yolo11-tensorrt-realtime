# Deployment Guide

## Compatibility Checks

Before installing TensorRT, record the GPU model, NVIDIA driver, CUDA runtime used by PyTorch, and
the TensorRT/Python compatibility matrix. TensorRT engines are not portable across all GPUs or
TensorRT versions; rebuild from ONNX on the target machine.

```powershell
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import tensorrt as trt; print(trt.__version__)"
python -c "from cuda.bindings import runtime; print(runtime.cudaRuntimeGetVersion())"
```

The basic package must remain usable when either TensorRT or `cuda-python` is absent. The project
does not automatically install or change the system CUDA toolkit.

## Pipeline

1. Obtain official `yolo11s.pt` or a user-trained compatible detection weight.
2. Export static 640 x 640 ONNX and validate the file.
3. Verify ONNX Runtime output shape and inference.
4. Build the engine on the target TensorRT/GPU environment.
5. Run one TensorRT image inference before benchmarking.
6. Benchmark with warmup and synchronized streams.

## Failure Boundaries

- A missing weight may require first-run internet access or a local `--model` path.
- A missing CUDA Execution Provider is not an ONNX CPU failure.
- A missing TensorRT install is `SKIP`, not a test failure.
- An ONNX parser error should be resolved at export/model compatibility, not hidden.
- FP16 build fails explicitly if the platform does not report fast FP16 support.

