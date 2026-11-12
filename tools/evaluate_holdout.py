"""Evaluate deployed inference on a fixed upstream test subset, without model selection."""
from pathlib import Path
import csv
import io
import json
import sys
import zipfile
import xml.etree.ElementTree as E
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from inspect_reference import ROOT,request
from pingpong.detector import analyze_video


def main():
    folder=ROOT/'datasets/holdout25';folder.mkdir(parents=True,exist_ok=True)
    tree=E.fromstring(request('25/rallies_videos/',True))
    videos=sorted(Path(n.findtext('d:href',namespaces={'d':'DAV:'})).name for n in tree
                  if n.findtext('d:href',namespaces={'d':'DAV:'}).endswith('.mp4'))[:6]
    with zipfile.ZipFile(ROOT/'work/reference/all_csv_midpoint_annotations.zip') as z:
        for video in videos:
            data=z.read(f'all_csv_annotations/25_csv_{Path(video).stem}.csv').decode().replace('\r\r\n','\n')
            (folder/Path(video).with_suffix('.csv')).write_text(data,encoding='utf-8')
    def fetch(video):
        target=folder/video
        if not target.exists():target.write_bytes(request('25/rallies_videos/'+video))
        print('test sample',video,flush=True)
    with ThreadPoolExecutor(3) as pool:list(pool.map(fetch,videos))
    clip_results=[];tp=fp=fn=tn=0;errors=[];frames=0
    for video in videos:
        path=folder/video
        with path.with_suffix('.csv').open() as f:labels=list(csv.DictReader(f))
        project=analyze_video(path,ROOT/'models/detector.onnx')
        if project['frame_count']!=len(labels):raise RuntimeError('Holdout alignment mismatch: '+video)
        counts=dict(tp=0,fp=0,fn=0,tn=0)
        for p,label in zip(project['tracks'],labels):
            if p['frame']!=int(label['Frame']):raise RuntimeError('Holdout frame index mismatch')
            visible=label['Visibility']=='1';found=p['x'] is not None
            if visible and found:
                error=float(np.hypot((p['x']-float(label['X']))*512/project['width'],
                                     (p['y']-float(label['Y']))*288/project['height']))
                errors.append(error)
                if error<=6:counts['tp']+=1
                else:counts['fp']+=1;counts['fn']+=1
            elif visible:counts['fn']+=1
            elif found:counts['fp']+=1
            else:counts['tn']+=1
        tp+=counts['tp'];fp+=counts['fp'];fn+=counts['fn'];tn+=counts['tn'];frames+=len(labels)
        clip_results.append(dict(file=video,frames=len(labels),**counts,performance=project['performance']))
        print('evaluated',video,counts,flush=True)
    precision=tp/max(1,tp+fp);recall=tp/max(1,tp+fn)
    result=dict(dataset='BlurBall official dataset, match group 25, lexicographically first 6 clips',
        upstream_split='25 is listed in upstream dataset config test.matches; unused in local fine-tuning/selection',
        clips=clip_results,frames=frames,tp=tp,fp=fp,fn=fn,tn=tn,threshold=.5,
        matching_radius_512px=6,precision=precision,recall=recall,
        f1=2*precision*recall/max(1e-9,precision+recall),
        mean_error_on_detected_visible_frames_512px=float(np.mean(errors)) if errors else None,
        caveats=['Small upstream test subset, not full-dataset or contest accuracy.',
                 'Training membership of published weights cannot be independently audited.',
                 'Ball localization only; no bounce ground truth exists in these CSV files.'])
    (ROOT/'reports/holdout_evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
