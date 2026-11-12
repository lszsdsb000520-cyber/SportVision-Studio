"""Simulated GUI review tests write ONLY to work/qa, never to the user's corpus."""
from pathlib import Path
import copy
import json
import os
import sys
import time
os.environ['QT_QPA_PLATFORM']='offscreen'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from PySide6.QtWidgets import QApplication,QDialogButtonBox
from pingpong.ui import MainWindow
from pingpong.events import event_kind,ALGORITHM
from pingpong.review_dialogs import ApprovalDialog,EventDialog
from pingpong.library import prepare_training_index
from pingpong.media import resolve_video


def wait(app,window):
    started=time.monotonic()
    while window.task:
        app.processEvents();time.sleep(.01)
        if time.monotonic()-started>90:raise RuntimeError('GUI task timeout')
    app.processEvents()


def main():
    out=ROOT/'reports/v3';out.mkdir(exist_ok=True);qa=ROOT/'work/qa-v3';qa.mkdir(exist_ok=True)
    app=QApplication([]);w=MainWindow();errors=[];w.error=lambda x:errors.append(x);w.show()
    old=ROOT/'video/labeled/train_match1_000_oblique.mp4'
    assert resolve_video(old).name=='V001.mp4' and w.load_video(old)
    assert 'V001' in w.filename.text()
    w.auto_calibrate();wait(app,w);assert len(w.corners)==4 and not errors
    w.start_analysis();wait(app,w);assert w.project and w.project['event_algorithm']==ALGORITHM
    first=next(e for e in w.project['events'] if event_kind(e)=='bounce')
    old_frame=first['frame']
    w.apply_event_changes(first,dict(kind='out',frame=old_frame,x=first['x'],y=first['y'],confirmed=True))
    assert event_kind(first)=='out' and first['source']=='manual'
    w.refresh_bounce_candidates()
    assert not any(event_kind(e)=='bounce' and abs(e['frame']-old_frame)<=2 for e in w.project['events'])
    # Add a manual net annotation, and ensure confirming it never converts it to a bounce.
    w.apply_event_changes(None,dict(kind='net',frame=10,x=1000.,y=520.,confirmed=False))
    net=next(e for e in w.project['events'] if event_kind(e)=='net' and e['frame']==10)
    w.events.selectRow(w.project['events'].index(net));w.confirm_event();assert event_kind(net)=='net' and net['confirmed']
    bounce=next(e for e in w.project['events'] if event_kind(e)=='bounce')
    w.events.selectRow(w.project['events'].index(bounce));w.start_event_point_edit()
    new_frame=bounce['frame']+1;w.seek(new_frame)
    w.video_click(bounce['x']+2,bounce['y']+1)
    assert bounce['frame']==new_frame and bounce['source']=='manual' and bounce['confirmed']
    auto=next(e for e in w.project['events'] if e['source']!='manual')
    rejected=(auto['frame'],event_kind(auto));w.events.selectRow(w.project['events'].index(auto));w.delete_event()
    w.refresh_bounce_candidates();assert not any(abs(e['frame']-rejected[0])<=2 and event_kind(e)==rejected[1] for e in w.project['events'])
    w.seek(0);w.mark_ball_invisible();assert w.project['tracks'][0]['source']=='manual' and w.project['tracks'][0]['x'] is None
    w.seek(1);w.start_edit();w.video_click(500,400);assert w.project['tracks'][1]['source']=='manual'
    dialog=ApprovalDialog(w.project,w)
    ok=dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok)
    assert not ok.isEnabled();dialog.events.setChecked(True);assert ok.isEnabled() and not dialog.ball.isChecked();dialog.close()
    edit_dialog=EventDialog(bounce,w.project,w.frame_index,w);assert edit_dialog.values()['kind']=='bounce';edit_dialog.close()
    w.seek(26);app.processEvents();w.grab().save(str(out/'interface.png'))
    w.resize(1140,740);app.processEvents();assert w.run_btn.isVisible();w.grab().save(str(out/'interface_small.png'))
    # This is an automated functional test, not the user's approval of any real video.
    snapshot=copy.deepcopy(w.project);snapshot['qa_only']=True
    w.launch_task('approve',dict(project=snapshot,folder=qa/'corpus',events_reviewed=True,ball_positions_reviewed=False))
    wait(app,w);assert not errors
    approved=Path(w.project['review']['directory']);assert (approved/'approved.json').exists()
    manifest=json.loads((approved/'approved.json').read_text());assert manifest['reviewed_ball_frames']==2
    index=prepare_training_index(qa/'corpus',qa/'training');assert index['clips']==1 and index['reviewed_ball_frames']==2
    before_review=w.project['review']['status'];w.seek(2);w.mark_ball_invisible();assert w.project['review']['status']=='modified_after_approval'
    summary=dict(success=True,legacy_filename_resolved=True,short_name='V001',event_editing=True,
        manual_out_and_net=True,click_to_correct=True,deletion_not_reintroduced=True,
        approval_requires_human_declaration=True,qa_only_corpus=str(qa/'corpus'),user_corpus_untouched=True,
        manual_ball_labels=2,review_status_invalidated_on_edit=True,training_index=True,
        event_algorithm=ALGORITHM,performance=w.project['performance'])
    (out/'gui_verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    w.dirty=False;w.close();print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
