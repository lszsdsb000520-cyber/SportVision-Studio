"""Actual widget, automatic-boundary, frame navigation and bounce review checks."""
from pathlib import Path
import os
import sys
import json
import time
os.environ['QT_QPA_PLATFORM']='offscreen'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFontMetrics,QPalette
from pingpong.ui import MainWindow
from pingpong.analysis import TrackPoint,propose_bounces,export_csv


def wait(app,window):
    started=time.monotonic()
    while window.task:
        app.processEvents();time.sleep(.01)
        if time.monotonic()-started>90:raise RuntimeError('task timed out')
    app.processEvents()


def main():
    app=QApplication([]);window=MainWindow();errors=[];window.error=lambda e:errors.append(e)
    window.show();out=ROOT/'reports/v2'
    if os.environ.get('QT_SCALE_FACTOR')=='1.5':out=out/'scaled150'
    out.mkdir(parents=True,exist_ok=True)
    assert window.load_video(ROOT/'sample/demo.mp4')
    window.auto_calibrate();wait(app,window)
    assert len(window.corners)==4 and not errors
    corners=[p[:] for p in window.corners]
    for step in (2,3,7):
        window.seek(0);window.step_input.setValue(step);window.step_frames(1)
        assert window.frame_index==step
        window.step_frames(-1);assert window.frame_index==0
    window.seek(198);window.step_frames(1);assert window.frame_index==198
    window.goto_input.setValue(26);window.goto_btn.click();assert window.frame_index==26
    window.drag_corner(2,corners[2][0]+4,corners[2][1]+2,False)
    window.drag_corner(2,corners[2][0]+4,corners[2][1]+2,True)
    assert abs(window.corners[2][0]-corners[2][0]-4)<.01
    window.apply_corners(corners,{'method':'automatic_verified'})
    window.start_analysis();wait(app,window);assert window.project and not errors
    frames=[e['frame'] for e in window.project['events']]
    assert 23 in frames and 22 not in frames
    event=next(e for e in window.project['events'] if e['frame']==23)
    p=window.project['tracks'][23];assert event['x']==p['x'] and event['y']==p['y']
    row=next(i for i,e in enumerate(window.project['events']) if e['frame']==23)
    window.events.selectRow(row);window.confirm_event();assert window.project['events'][row]['confirmed']
    window.apply_corners(corners);assert any(e['frame']==23 and e['confirmed'] for e in window.project['events'])
    window.seek(26);window.resize(1440,920);app.processEvents();window.grab().save(str(out/'interface.png'))
    window.resize(1140,740);app.processEvents()
    assert window.run_btn.isVisible() and window.cancel_btn.isVisible()
    assert window.run_btn.geometry().bottom()<=window.run_btn.parentWidget().height()
    window.grab().save(str(out/'interface_small.png'))
    window.side_scroll.verticalScrollBar().setValue(window.side_scroll.verticalScrollBar().maximum())
    app.processEvents();assert window.export_btn.isVisible()
    window.grab().save(str(out/'interface_scrolled.png'))
    window.acceleration.showPopup();app.processEvents()
    window.acceleration.view().window().grab().save(str(out/'dropdown.png'));window.acceleration.hidePopup()
    assert app.style().objectName().lower()=='fusion'
    assert QFontMetrics(app.font()).inFont('乒') and QFontMetrics(app.font()).inFont('轨')
    assert app.palette().color(QPalette.HighlightedText).name()=='#ffffff'
    export_csv(out/'sample_result',window.project)
    baseline=json.loads((ROOT/'reports/frozen-final/project.json').read_text(encoding='utf-8'))
    updated=propose_bounces([TrackPoint(**p) for p in baseline['tracks']],baseline['fps'])
    result=dict(success=True,automatic_corners=corners,new_sample_events=frames,
        previous_events=[e['frame'] for e in baseline['events']],updated_unfiltered_events=[e['frame'] for e in updated],
        frame_steps=[2,3,7],absolute_frame_jump=True,draggable_corners=True,reviewed_events_preserved=True,
        compact_layout=[1140,740],main_actions_remain_visible=True,style=app.style().objectName(),
        chinese_font=app.font().family(),performance=window.project['performance'])
    (out/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    window.dirty=False;window.close();print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
