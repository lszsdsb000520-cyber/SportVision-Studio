"""Inference backends compatible with the official image and local Windows."""
import os
from pathlib import Path


def model_runner(path, context, name, shape, *, use_trt=True):
    if use_trt and os.name != "nt":
        try:
            from core.trt import load_or_build
            model, status = load_or_build(path, Path(context.output_dir)/"trt_cache",
                                          input_name=name, shape=shape)
            return model.infer, "tensorrt_"+status
        except Exception as exc:
            print("TensorRT unavailable:", exc)
    import onnxruntime as ort
    available = ort.get_available_providers()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    device = os.getenv("DML_ID", "0")
    providers = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    if "DmlExecutionProvider" in available and device != "cpu":
        opts.enable_mem_pattern = False
        providers.append(("DmlExecutionProvider", {"device_id": device}))
    providers.append("CPUExecutionProvider")
    try:
        session = ort.InferenceSession(str(path),opts,providers=providers)
    except Exception as exc:
        if len(providers) == 1:
            raise
        print("GPU ONNX unavailable:", exc)
        session = ort.InferenceSession(str(path),opts,providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    return lambda x: session.run(None,{name:x})[0],session.get_providers()[0]
