from pathlib import Path
import json,sys,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pingpong.detector import analyze_video,render_frame
from pingpong.analysis import save_project
import cv2


def main():
    out=ROOT/'reports/v6';before=json.loads((out/'V009.before.json').read_text(encoding='utf-8'))
    after=analyze_video(ROOT/'video/labeled/V009.mp4',ROOT/'models/detector.onnx')
    save_project(out/'V009.after.json',after)
    def artifact(p):return p['x'] is not None and np.hypot(p['x']-1022,p['y']-650)<15
    a=[p['frame'] for p in before['tracks'] if p['frame']>=200 and artifact(p)]
    b=[p['frame'] for p in after['tracks'] if p['frame']>=200 and artifact(p)]
    report=dict(before_artifact_frames=a,after_artifact_frames=b,rejected=len(after['motion_rejections']),
                scores=[(p['frame'],p['motion_score']) for p in after['tracks'][205::3]])
    (out/'stationary_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)
    assert a and not b,report
    cap=cv2.VideoCapture(after['video']);cap.set(cv2.CAP_PROP_POS_FRAMES,214);ok,frame=cap.read();cap.release()
    if ok:cv2.imwrite(str(out/'V009.after.jpg'),cv2.resize(render_frame(frame,after,214),(960,540)))


if __name__=='__main__':main()
