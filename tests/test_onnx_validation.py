from pathlib import Path

import onnx
from onnx import TensorProto, helper

from yolo11_deploy.onnx_validation import validate_onnx_artifact


def test_validate_onnx_artifact_reports_graph_interface(tmp_path: Path) -> None:
    input_info = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 8, 8])
    output_info = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 3, 8, 8])
    graph = helper.make_graph(
        [helper.make_node("Identity", ["images"], ["output0"])],
        "test",
        [input_info],
        [output_info],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    metadata = model.metadata_props.add()
    metadata.key = "names"
    metadata.value = "{0: 'object'}"
    path = tmp_path / "model.onnx"
    onnx.save(model, path)
    info = validate_onnx_artifact(path)
    assert info.opset == 17
    assert info.inputs == (("images", (1, 3, 8, 8)),)
    assert info.outputs == (("output0", (1, 3, 8, 8)),)
    assert info.class_count == 1
