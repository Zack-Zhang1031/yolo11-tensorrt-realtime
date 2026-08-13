"""ONNX artifact validation and human-readable model inspection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ONNXArtifactInfo:
    """Validated ONNX deployment metadata."""

    path: Path
    size_bytes: int
    opset: int
    inputs: tuple[tuple[str, tuple[int | str, ...]], ...]
    outputs: tuple[tuple[str, tuple[int | str, ...]], ...]
    class_count: int | None


def validate_onnx_artifact(model_path: str | Path) -> ONNXArtifactInfo:
    """Load an ONNX artifact, run the checker, and summarize its graph interface."""
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("ONNX is required to validate exported models") from exc
    path = Path(model_path).expanduser().resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Non-empty ONNX model not found: {path}")
    model = onnx.load(str(path), load_external_data=True)
    onnx.checker.check_model(model)
    opset = max((item.version for item in model.opset_import), default=0)
    metadata = {entry.key: entry.value for entry in model.metadata_props}
    return ONNXArtifactInfo(
        path=path,
        size_bytes=path.stat().st_size,
        opset=opset,
        inputs=tuple((value.name, _value_shape(value)) for value in model.graph.input),
        outputs=tuple((value.name, _value_shape(value)) for value in model.graph.output),
        class_count=_class_count(metadata.get("names")),
    )


def _value_shape(value: Any) -> tuple[int | str, ...]:
    dimensions: list[int | str] = []
    for dimension in value.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            dimensions.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            dimensions.append(str(dimension.dim_param))
        else:
            dimensions.append("dynamic")
    return tuple(dimensions)


def _class_count(raw_names: str | None) -> int | None:
    if not raw_names:
        return None
    import ast

    try:
        names = ast.literal_eval(raw_names)
    except (SyntaxError, ValueError):
        return None
    return len(names) if isinstance(names, (dict, list, tuple)) else None
