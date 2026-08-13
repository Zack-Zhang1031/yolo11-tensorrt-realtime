# Benchmark Methodology

## Standard Conditions

- Input: 640 x 640
- Batch: 1
- Warmup: 50 calls
- Measured runs: 200
- Statistics: arithmetic mean, median/P50, P95, and FPS

The benchmark scripts measure raw model/backend inference. Image decode, letterbox preprocessing,
NMS, drawing, display, and video encoding are excluded. State that distinction beside published
numbers.

## Synchronization

CUDA work is asynchronous. PyTorch CUDA benchmarks call `torch.cuda.synchronize()` around timed
inference. TensorRT copies and inference run on a CUDA stream and synchronize before the end time.
ONNX Runtime calls are synchronous at the Python API boundary for the configured providers used by
this script.

## Reporting

Only report measurements emitted by an executed command, with hardware and software context.
Unavailable backends are `N/A` or `SKIP`. Historical values in the README are clearly separated
from measurements reproduced in the current checkout.

