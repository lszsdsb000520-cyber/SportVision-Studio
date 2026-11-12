"""Exercise the actual GUI worker and export path with real inference."""
from pathlib import Path
import json
import os
import sys
import time
os.environ['QT_QPA_PLATFORM']='offscreen'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFontDatabase,QFont
from pingpong.ui import MainWindow
from pingpong.analysis import export_csv,save_project,validate_project
from pingpong.detector import export_video


def wait(app,window,timeout=90):
    started=time.monotonic();ticks=0
    while window.task:
        app.processEvents();time.sleep(.01);ticks+=1
        if time.monotonic()-started>timeout:raise RuntimeError('GUI task timeout')
    app.processEvents();return ticks


def main():
    app=QApplication([])
    for name in ('msyh.ttc','msyhbd.ttc'):QFontDatabase.addApplicationFont('C:/Windows/Fonts/'+name)
    app.setFont(QFont('Microsoft YaHei UI',10))
    window=MainWindow();window.show();app.processEvents()
    video=ROOT/'video/labeled/train_match1_000_oblique.mp4'
    if not video.exists():video=ROOT/'sample/demo.mp4'
    assert window.load_video(video)
    window.start_calibration()
    for xy in ([1380,550],[980,500],[555,557],[714,665]):window.video_click(*xy)
    assert len(window.corners)==4 and not window.calibrating
    window.start_analysis();ticks=wait(app,window)
    assert window.project and window.run_btn.isEnabled() and ticks>10
    project=window.project;validate_project(project)
    assert project['frame_count']==199 and project['events']
    window.events.selectRow(0);window.confirm_event()
    assert project['events'][0]['confirmed']
    window.event_clicked(0,0);assert window.frame_index==project['events'][0]['frame']
    window.start_edit();p=project['tracks'][window.frame_index];window.video_click(p['x'],p['y'])
    assert project['tracks'][window.frame_index]['source']=='manual'
    output=ROOT/'reports/smoke';output.mkdir(parents=True,exist_ok=True)
    export_csv(output,project)
    reloaded=json.loads((output/'project.json').read_text(encoding='utf-8'));validate_project(reloaded)
    assert reloaded['events'][0]['confirmed']
    assert export_video(output/'demo_tracked.mp4',project)
    window.seek(88);window.refresh_events();app.processEvents()
    window.grab().save(str(output/'ui.png'))
    # A cancelled new run must retain the previous reviewed project.
    window.dirty=False;old=window.project;window.start_analysis();window.cancel_task();wait(app,window)
    assert window.project is old and window.run_btn.isEnabled()
    result=dict(gui_main_loop_iterations=ticks,frames=project['frame_count'],
        events=len(project['events']),performance=project['performance'],
        checks=['real ONNX analysis','GUI remains responsive','four-corner calibration',
                'event seek','event confirmation','manual ball correction','CSV/JSON roundtrip',
                'video export','cancellation retains reviewed project'])
    (output/'smoke_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    window.dirty=False;window.close();print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
