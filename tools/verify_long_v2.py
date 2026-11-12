"""Real 30-second inputs: inference, automatic table detection, and cut handling."""
from pathlib import Path
import json
import sys
import cv2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pingpong.boundary import TableBoundaryDetector
from pingpong.detector import analyze_video
from pingpong.analysis import validate_project
from pingpong.media import resolve_video


def main():
    root=Path(__file__).resolve().parents[1];folder=root/'video/by_duration/00_30s_tests'
    dest=folder/'v2_analysis';dest.mkdir(exist_ok=True)
    table=TableBoundaryDetector();results=[]
    for camera in ('oblique','side'):
        video=resolve_video(folder/f'test_30s_{camera}_compilation.mp4')
        cap=cv2.VideoCapture(str(video));ok,frame=cap.read();cap.release();assert ok
        calibration=table.detect(frame)
        project=analyze_video(video,root/'models/detector.onnx',corners=calibration['corners'])
        project['calibration_info']=calibration;validate_project(project)
        assert len(project['break_frames'])==2
        assert not any(abs(e['frame']-cut)<=2 for e in project['events'] for cut in project['break_frames'])
        target=dest/(video.stem+'.analysis.json');target.write_text(json.dumps(project,indent=2),encoding='utf-8')
        record=dict(video=video.name,frames=project['frame_count'],seconds=project['frame_count']/project['fps'],
                    automatic_corners=project['corners'],break_frames=project['break_frames'],
                    bounce_candidates=len(project['events']),performance=project['performance'],project=str(target))
        results.append(record);print(json.dumps(record,indent=2),flush=True)
    (root/'reports/v2/long_video_verification.json').write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__=='__main__':main()
