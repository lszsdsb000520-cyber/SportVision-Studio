from pathlib import Path
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import copy,json,sys,time,tempfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from PySide6.QtCore import Qt,QPointF,QEvent,QTimer
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication
from pingpong.ui import MainWindow
from pingpong.analysis import TrackPoint,save_project
from pingpong.events import analyze_events
from pingpong.detector import analyze_video


def wait(app,w):
    started=time.monotonic()
    while w.task:
        app.processEvents();time.sleep(.005)
        if time.monotonic()-started>15:raise RuntimeError('Cancellation failed to finish')


def mouse(view,kind,p,button,buttons):QApplication.sendEvent(view,QMouseEvent(kind,p,p,button,buttons,Qt.NoModifier))


def main():
    out=ROOT/'reports/v7';app=QApplication([]);w=MainWindow();errors=[];w.error=errors.append;w.show();app.processEvents()
    before=json.loads((out/'V011.before.json').read_text(encoding='utf-8'))
    after=analyze_video(before['video'],ROOT/'models/detector.onnx',corners=before['corners'])
    save_project(out/'V011.after.json',after)
    event=next(e for e in after['events'] if abs(e['frame']-209)<=1)
    assert event['kind']=='stroke' and event['table_xy'] is None
    assert any(e['kind']=='bounce' and abs(e['frame']-197)<=1 for e in after['events'])
    assert w.load_video(after['video']);w.project=after;w.corners=copy.deepcopy(after['corners']);w.refresh_events();w.refresh_frame();w.update_controls()
    w.view.zoom_at(2);w.pan_button.setChecked(True);old_pan=QPointF(w.view.pan);old_corners=copy.deepcopy(w.corners)
    mouse(w.view,QEvent.MouseButtonPress,QPointF(300,140),Qt.LeftButton,Qt.LeftButton)
    mouse(w.view,QEvent.MouseMove,QPointF(360,170),Qt.NoButton,Qt.LeftButton)
    mouse(w.view,QEvent.MouseButtonRelease,QPointF(360,170),Qt.LeftButton,Qt.NoButton)
    assert w.view.pan!=old_pan and w.corners==old_corners
    w.pan_button.setChecked(False);w.view.allow_drag=False;old_pan=QPointF(w.view.pan)
    mouse(w.view,QEvent.MouseButtonPress,QPointF(300,140),Qt.LeftButton,Qt.LeftButton)
    mouse(w.view,QEvent.MouseMove,QPointF(330,145),Qt.NoButton,Qt.LeftButton)
    mouse(w.view,QEvent.MouseButtonRelease,QPointF(330,145),Qt.LeftButton,Qt.NoButton)
    assert w.view.pan!=old_pan
    original=copy.deepcopy(w.project)
    template=next(e for e in w.project['events'] if e['kind']=='bounce')
    w.project['events']=[dict(copy.deepcopy(template),frame=10+i*2,time=(10+i*2)/w.fps,event_number=i+1,table_xy=[.7,1.3]) for i in range(80)]
    w.selected_number=None;w.refresh_events();w.refresh_frame();app.processEvents();app.processEvents()
    assert len(w.table.label_rects)==80 and w.table.minimumHeight()>=870
    w.project=original;w.selected_number=None;w.refresh_events();w.refresh_frame();w.dirty=False
    old_project=w.project;started=time.monotonic();cancel_feedback=[]
    w.start_analysis()
    def stop():
        w.cancel_btn.click();cancel_feedback.append(w.cancel_btn.text())
    QTimer.singleShot(80,stop);wait(app,w)
    assert w.project is old_project and w.run_btn.isEnabled() and cancel_feedback==['取消中…']
    cancellation_seconds=time.monotonic()-started
    with tempfile.TemporaryDirectory(dir=ROOT/'work') as temporary:
        target=Path(temporary)/'cancel.mp4'
        w.launch_task('sync',dict(path=target,project=copy.deepcopy(w.project),height=480))
        QTimer.singleShot(80,w.cancel_task);wait(app,w)
        assert not target.exists() and not list(Path(temporary).glob('*.partial.mp4'))
    w.view.reset_zoom();w.select_number(event['event_number'] if 'event_number' in event else next(e['event_number'] for e in w.project['events'] if e['kind']=='stroke'))
    w.resize(1440,920);app.processEvents();w.grab().save(str(out/'interface.png'))
    w.resize(1120,740);app.processEvents();assert w.width()<=1120;w.grab().save(str(out/'small.png'))
    report=dict(success=True,V011_209_kind='stroke',V011_197_bounce_preserved=True,
        explicit_left_pan=True,default_left_pan=True,all_80_dense_numbers=True,
        cancel_preserves_previous_analysis=True,cancel_seconds=cancellation_seconds,cancel_export_cleanup=True,errors=errors)
    assert not errors,errors
    (out/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report));w.dirty=False;w.close()


if __name__=='__main__':main()
