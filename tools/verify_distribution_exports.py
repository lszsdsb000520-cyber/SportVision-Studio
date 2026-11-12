from pathlib import Path
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import json,sys,cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pingpong.distribution_export import export_distribution,export_synchronized,visible_bounces


def main():
    project=json.loads((ROOT/'reports/v5/frozen/project.json').read_text(encoding='utf-8'))
    project['video']=str(ROOT/'sample/demo.mp4')
    output=ROOT/'reports/v6';output.mkdir(exist_ok=True)
    first=project['events'][0];number=first['event_number']
    assert not visible_bounces(project,index=first['frame']-1)
    assert len(visible_bounces(project,index=first['frame']))==1
    assert all(e['event_number']!=number for e in visible_bounces(project,{'hidden_event_numbers':[number]}))
    export_distribution(output/'sample_distribution.png',project)
    export_synchronized(output/'sample_synchronized.mp4',project,height=720)
    cap=cv2.VideoCapture(str(output/'sample_synchronized.mp4'))
    frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);width=int(cap.get(3));height=int(cap.get(4))
    assert frames==project['frame_count'] and abs(fps-project['fps'])<.01 and (width,height)==(1680,720)
    cap.set(cv2.CAP_PROP_POS_FRAMES,first['frame']);ok,image=cap.read();cap.release();assert ok
    cv2.imwrite(str(output/'synchronized_preview.jpg'),image)
    target=output/'cancelled_sync.mp4'
    assert export_synchronized(target,project,cancelled=lambda:True) is None
    assert not target.exists() and not target.with_name(target.stem+'.partial.mp4').exists()
    result=dict(success=True,plot=True,synchronized_video=True,frames=frames,fps=fps,
                frame_synchronous_points=True,hidden_points_respected=True,cancellation_cleanup=True)
    (output/'export_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
