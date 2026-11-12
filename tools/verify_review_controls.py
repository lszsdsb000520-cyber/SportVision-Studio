"""GUI checks use temporary QA files, never the human-approved library."""
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
from pathlib import Path
import sys,json,copy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from PySide6.QtWidgets import QApplication,QDialogButtonBox
from pingpong.ui import MainWindow
from pingpong.review_dialogs import InvisibleRangeDialog
from pingpong.library import save_approved


def main():
    app=QApplication([]);w=MainWindow();w.show();app.processEvents()
    assert w.load_video(ROOT/'sample/demo.mp4')
    w.project=json.loads((ROOT/'reports/v3.3/frozen/project.json').read_text(encoding='utf-8'))
    w.project['video']=w.video;w.corners=copy.deepcopy(w.project['corners'])
    w.refresh_events();w.update_controls();w.seek(23)
    original=copy.deepcopy(w.project)
    w.set_range_marker(True);w.seek(25);w.set_range_marker(False)
    dialog=InvisibleRangeDialog(w.total,w.frame_index,w.range_start,w.range_end,w)
    assert dialog.start.value()==23 and dialog.end.value()==25 and '3' in dialog.summary.text()
    dialog.start.setValue(27)
    assert not dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok).isEnabled()
    w.apply_invisible_range(23,25)
    assert all(p['x'] is None for p in w.project['tracks'][23:26])
    assert w.project['events']==original['events']
    w.undo_annotation();assert w.project['tracks']==original['tracks']
    w.apply_invisible_range(23,25,True);assert not any(23<=e['frame']<=25 for e in w.project['events'])
    w.undo_annotation();assert w.project['events']==original['events']
    for key,check in w.layer_checks.items():
        check.setChecked(False);assert w.display_options()[key] is False;check.setChecked(True)
    w.clean_view.setChecked(True);assert not any(w.display_options().values())
    assert not w.view.allow_drag and not w.table.show_points
    w.clean_view.setChecked(False);assert all(w.display_options().values())
    w.seek(0);w.jump_event(1);assert w.frame_index==23
    w.jump_event(1);assert w.frame_index==48
    w.jump_event(-1);assert w.frame_index==23
    w.layer_checks['boundary'].setChecked(False);assert not w.view.allow_drag
    w.start_calibration();assert w.layer_checks['boundary'].isChecked();w.cancel_annotation()
    w.seek(26);w.resize(1440,920);app.processEvents()
    out=ROOT/'reports/v3.4';out.mkdir(exist_ok=True)
    w.grab().save(str(out/'app.png'))
    w.resize(1120,740);app.processEvents();w.grab().save(str(out/'small.png'))
    assert w.width()<=1120
    w.apply_invisible_range(23,25)
    accepted=save_approved(w.project,ROOT/'work/qa-v3.4',events_reviewed=True)
    assert accepted['manifest']['reviewed_ball_frames']==3
    assert accepted['manifest']['video_file']=='demo_success.mp4'
    report=dict(success=True,inclusive_range=True,undo=True,independent_layers=True,
        event_navigation=True,small_window=True,success_filename_preserved=True,qa_only=True)
    (out/'controls.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report));w.dirty=False;w.close()


if __name__=='__main__':main()
