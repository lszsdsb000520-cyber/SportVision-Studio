from pathlib import Path
import sys
import argparse
import json
import numpy as np
import torch
import onnxruntime as ort
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pingpong.model import load_model, ExportModel


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, default=root/'models/blurball_pretrained.pt')
    parser.add_argument('--output', type=Path, default=root/'models/detector.onnx')
    args = parser.parse_args()
    torch.set_num_threads(4)
    model = ExportModel(load_model(root,args.weights)).eval()
    x = torch.randn(1,9,288,512)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        reference = model(x).numpy()
        torch.onnx.export(model,x,str(args.output),input_names=['frames'],output_names=['heatmaps'],
                          opset_version=17,dynamo=False)
    options=ort.SessionOptions(); options.intra_op_num_threads=4
    session = ort.InferenceSession(str(args.output),options,providers=['CPUExecutionProvider'])
    actual=session.run(None,{'frames':x.numpy()})[0]
    error=float(np.max(np.abs(actual-reference)))
    if not np.allclose(actual,reference,atol=1e-3,rtol=1e-3):
        raise RuntimeError(f'ONNX numerical mismatch: {error}')
    report=dict(weights=str(args.weights),onnx=str(args.output),shape=list(actual.shape),max_abs_error=error)
    (args.output.with_suffix('.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
