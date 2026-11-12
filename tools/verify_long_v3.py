from pathlib import Path
import json
import sys
import cv2
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pingpong.detector import analyze_video
from pingpong.boundary import TableBoundaryDetector


def main():
    root=Path(__file__).resolve().parents[1];folder=root/'video/by_duration/00_30s_tests'
    dest=folder/'v3_analysis';dest.mkdir(exist_ok=True);table=TableBoundaryDetector();results=[]
    for ident in ('T01','T02'):
        file=folder/(ident+'.mp4');cap=cv2.VideoCapture(str(file));ok,frame=cap.read();cap.release();assert ok
        calibration=table.detect(frame)
        p=analyze_video(file,root/'models/detector.onnx',corners=calibration['corners'])
        p['calibration_info']=calibration
        (dest/(ident+'.analysis.json')).write_text(json.dumps(p,indent=2),encoding='utf-8')
        old=json.loads((folder/'v2_analysis'/(ident+'.analysis.json')).read_text(encoding='utf-8'))
        row=dict(video=ident,frames=p['frame_count'],previous_bounces=len(old['events']),
            event_counts=dict(Counter(e['kind'] for e in p['events'])),
            recovered_ball_frames=p['recovered_ball_frames'],review_hints=len(p['review_hints']),
            performance=p['performance'],project=str(dest/(ident+'.analysis.json')),
            accuracy_claim=False,reason='All events remain candidates until human review.')
        results.append(row);print(json.dumps(row,indent=2),flush=True)
    (root/'reports/v3/long_video_verification.json').write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__=='__main__':main()
