"""Runtime-only validation, including DirectML numeric and task checks."""
from pathlib import Path
import argparse,json,sys,time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pingpong.detector import Detector,preprocess


def evaluate(detector,manifest,split,threshold):
    counts=dict(tp=0,fp=0,fn=0,tn=0);errors=[]
    for record in manifest['windows']:
        if record['split']!=split:continue
        images=[cv2.imread(str(ROOT/'datasets/extended'/p)) for p in record['images']]
        predictions=detector.predict(images,threshold)
        for prediction,name in zip(predictions,record['labels']):
            label=json.loads((ROOT/'datasets/extended'/name).read_text());x,y,_=prediction
            if label['visible'] and x is not None:
                error=float(np.hypot(x-label['x'],y-label['y']));errors.append(error)
                if error<=6:counts['tp']+=1
                else:counts['fp']+=1;counts['fn']+=1
            elif label['visible']:counts['fn']+=1
            elif x is not None:counts['fp']+=1
            else:counts['tn']+=1
    precision=counts['tp']/max(1,counts['tp']+counts['fp']);recall=counts['tp']/max(1,counts['tp']+counts['fn'])
    return dict(**counts,precision=precision,recall=recall,f1=2*precision*recall/max(1e-9,precision+recall),
                mean_error_512px=float(np.mean(errors)) if errors else None)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--threshold',type=float,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();manifest=json.loads((ROOT/'datasets/extended/manifest.json').read_text())
    detector=Detector(args.model)
    sample=np.load(args.model.with_suffix('.input.npy'));reference=np.load(args.model.with_suffix('.reference.npy'))
    actual=detector.session.run(None,{detector.input_name:sample})[0]
    max_relative=float((np.abs(actual-reference)/np.maximum(np.abs(reference),1e-6)).max())
    max_absolute=float(np.abs(actual-reference).max())
    assert max_relative<1e-3,(max_relative,max_absolute)
    task={split:evaluate(detector,manifest,split,args.threshold) for split in ('val','test')}
    for _ in range(5):detector.session.run(None,{detector.input_name:sample})
    times=[]
    for _ in range(30):
        start=time.perf_counter();detector.session.run(None,{detector.input_name:sample});times.append((time.perf_counter()-start)*1000)
    report=dict(success=True,provider=detector.provider,numerical_max_relative=max_relative,numerical_max_absolute=max_absolute,
        threshold=args.threshold,task=task,warmups=5,repetitions=30,threads=4,
        p50_ms_per_triplet=float(np.median(times)),p95_ms_per_triplet=float(np.percentile(times,95)),
        inference_fps=3000/float(np.mean(times)),vram='DirectML process peak not available under Windows WDDM; not inferred from PyTorch stats')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
