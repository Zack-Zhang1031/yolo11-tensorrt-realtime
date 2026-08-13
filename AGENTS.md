# Project Agent Rules

1. Do not train a model unless the user explicitly requests it.
2. Do not download large datasets.
3. Do not commit model weights or generated engines.
4. Do not fabricate benchmark measurements.
5. Do not fabricate mAP or other accuracy metrics.
6. Run `pytest` after changing the inference pipeline.
7. TensorRT changes must not make tests fail on machines without TensorRT.
8. Keep TensorRT an optional dependency and import it lazily.
9. Synchronize the GPU or CUDA stream around benchmark measurements.
10. Run `python scripts/smoke_test.py` before pushing.
11. Do not modify global Python site-packages.
12. Do not delete existing Git history.
13. Never force-push.

