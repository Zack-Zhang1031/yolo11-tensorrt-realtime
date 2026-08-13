# Benchmark Methodology

## Standard Conditions

- Input: 640 x 640
- Batch: 1
- Warmup: 50 calls
- Measured runs: 200
- Statistics: arithmetic mean, median/P50, P95, and FPS

The benchmark scripts exclude image decode, letterbox preprocessing, NMS, drawing, display, and
video encoding. PyTorch measures a forward pass with an input already resident on the selected
device. ONNX Runtime measures one session call. TensorRT measures host-to-device copy, engine
execution, and device-to-host output copy on its CUDA stream. State these boundaries beside
published numbers; the three values are useful backend measurements, not perfectly identical
microbenchmarks.

## Synchronization

CUDA work is asynchronous. PyTorch CUDA benchmarks call `torch.cuda.synchronize()` around timed
inference. TensorRT copies and inference run on a CUDA stream and synchronize before the end time.
ONNX Runtime calls are synchronous at the Python API boundary for the configured providers used by
this script.

## Reporting

Publish measurements together with the hardware, software versions, precision, input size, batch
size, warmup count, run count, and timing boundary. This makes results comparable and auditable.
