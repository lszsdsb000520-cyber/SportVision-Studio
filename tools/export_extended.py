"""Export bounded probability heatmaps with strict numerical checks on real inputs."""
from pathlib import Path
import argparse,hashlib,json,sys,time
import cv2,numpy as np,torch,onnx,onnxruntime as ort
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pingpong.model import load_model
from pingpong.detector import preprocess


class ProbabilityModel(torch.nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,frames):return torch.sigmoid(self.model(frames)[0]).clamp(.001,.999)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--threshold',type=float,default=.5)
    args=parser.parse_args();torch.set_num_threads(4)
    files=sorted((ROOT/'datasets/extended/images/val').glob('*.png'))
    assert len(files)>=12
    samples=[preprocess([cv2.imread(str(p)) for p in files[start:start+3]]) for start in (0,3,6,9)]
    model=ProbabilityModel(load_model(ROOT,args.weights)).eval();args.output.parent.mkdir(parents=True,exist_ok=True)
    np.save(args.output.with_suffix('.input.npy'),samples[0])
    with torch.inference_mode():
        torch.onnx.export(model,torch.from_numpy(samples[0]),str(args.output),input_names=['frames'],output_names=['heatmaps'],
                          opset_version=17,dynamo=False)
    graph=onnx.load(str(args.output));onnx.helper.set_model_props(graph,{'heatmap_activation':'sigmoid',
        'recommended_threshold':str(args.threshold),'input_format':'RGB_ImageNet_3frames_1x9x288x512',
        'probability_bounds':'0.001,0.999'})
    onnx.checker.check_model(graph);onnx.save(graph,str(args.output))
    options=ort.SessionOptions();options.intra_op_num_threads=4
    session=ort.InferenceSession(str(args.output),options,providers=['CPUExecutionProvider'])
    errors=[]
    for sample in samples:
        with torch.inference_mode():reference=model(torch.from_numpy(sample)).numpy()
        if not errors:np.save(args.output.with_suffix('.reference.npy'),reference)
        actual=session.run(None,{'frames':sample})[0]
        errors.append(dict(max_absolute=float(np.abs(actual-reference).max()),
            max_relative=float((np.abs(actual-reference)/np.maximum(np.abs(reference),1e-6)).max())))
    report=dict(weights=str(args.weights),onnx=str(args.output),shape=list(actual.shape),numerical_checks=errors,
        max_relative_error=max(e['max_relative'] for e in errors),max_absolute_error=max(e['max_absolute'] for e in errors),
        threshold=args.threshold,activation='sigmoid',probability_floor=.001,provider=session.get_providers()[0],
        onnx_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
        preprocessing='BGR decoded frames; bilinear resize512x288; RGB; ImageNet; NCHW concatenated 3frames')
    (args.output.with_suffix('.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)
    if report['max_relative_error']>=1e-3:raise RuntimeError('Numerical agreement below required tolerance')


if __name__=='__main__':main()
