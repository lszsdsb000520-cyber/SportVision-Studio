"""Exercise table-map linking, overlap selection, filters and non-destructive visibility."""
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
from pathlib import Path
import copy,csv,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from PySide6.QtCore import Qt,QPointF,QEvent,QItemSelectionModel
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication
from pingpong.ui import MainWindow
from pingpong.analysis import save_project,export_csv
from pingpong.library import save_approved
from pingpong.review_state import hidden_numbers


def click(view,point):
    QApplication.sendEvent(view,QMouseEvent(QEvent.MouseButtonPress,point,point,Qt.LeftButton,Qt.LeftButton,Qt.NoModifier))


def main():
    app=QApplication([]);w=MainWindow();errors=[];w.error=errors.append;w.show();app.processEvents()
    assert w.load_video(ROOT/'sample/demo.mp4')
    w.project=json.loads((ROOT/'reports/v4/frozen/project.json').read_text(encoding='utf-8'))
    w.project['video']=w.video;w.corners=copy.deepcopy(w.project['corners']);w.refresh_events();w.refresh_frame()
    w.update_controls();app.processEvents()
    one,two=w.project['events'][:2];n1,n2=one['event_number'],two['event_number']
    original_xy=copy.deepcopy(two['table_xy']);two['table_xy']=one['table_xy'][:];w.refresh_events()
    w.table.last_hits=();position=w.table.point_position(one)
    click(w.table,position);assert w.selected_number==n1 and w.frame_index==one['frame']
    click(w.table,position);assert w.selected_number==n2 and w.frame_index==two['frame']
    assert f'#{n2:03d}' in w.selection_details.text()
    two['table_xy']=original_xy;w.project['review']={'status':'approved'}
    before=copy.deepcopy(w.project['events']);w.hide_number(n2)
    assert n2 in hidden_numbers(w.project) and w.project['events']==before
    assert w.project['review']['status']=='approved'
    assert n2 not in [e['event_number'] for e in w.table.visible_events()]
    w.select_number(n1);w.confirm_event()
    w.event_filter.setCurrentIndex(w.event_filter.findData('pending'))
    target=w.project['events'][w.visible_rows[1]];target_number=target['event_number']
    w.events.setCurrentCell(1,0);w.confirm_event()
    assert w.event_filter.currentData()=='pending'
    assert next(e for e in w.project['events'] if e['event_number']==target_number)['confirmed']
    assert not next(e for e in w.project['events'] if e['event_number']==n2)['confirmed']
    w.event_filter.setCurrentIndex(w.event_filter.findData('confirmed'))
    row=next(row for row,index in enumerate(w.visible_rows) if w.project['events'][index]['event_number']==target_number)
    w.events.setCurrentCell(row,0);w.delete_event()
    assert target_number not in [e['event_number'] for e in w.project['events']]
    assert n1 in [e['event_number'] for e in w.project['events']]
    w.undo_annotation();assert target_number in [e['event_number'] for e in w.project['events']]
    w.clear_event_filters();w.select_number(n1)
    row2=next(row for row,index in enumerate(w.visible_rows) if w.project['events'][index]['event_number']==n2)
    w.events.selectionModel().select(w.events.model().index(row2,0),QItemSelectionModel.Select|QItemSelectionModel.Rows)
    assert len(w.selected_indices())==2
    w.toggle_selected_visibility();assert {n1,n2}<=hidden_numbers(w.project)
    assert len(w.selected_indices())==2
    w.restore_all_points();assert not hidden_numbers(w.project)
    w.event_search.setText(f'#{n2:03d}');assert len(w.visible_rows)==1
    w.clear_event_filters();w.hide_number(n2)
    qa=ROOT/'work/qa-v5';qa.mkdir(exist_ok=True)
    save_project(qa/'project.json',w.project)
    saved=json.loads((qa/'project.json').read_text(encoding='utf-8'))
    assert n2 in hidden_numbers(saved)
    export_csv(qa,w.project)
    with (qa/'events.csv').open(encoding='utf-8-sig') as file:rows=list(csv.DictReader(file))
    assert len(rows)==len(w.project['events']) and 'event_number' in rows[0]
    accepted=save_approved(w.project,qa/'corpus',events_reviewed=True)
    assert accepted['manifest']['events']==len(w.project['events'])
    w.restore_all_points();w.clear_event_filters();w.select_number(n1);w.resize(1440,920);app.processEvents()
    out=ROOT/'reports/v5';out.mkdir(exist_ok=True);w.grab().save(str(out/'interactive.png'))
    w.resize(1120,740);app.processEvents();assert w.width()<=1120
    w.grab().save(str(out/'small.png'))
    report=dict(success=True,overlap_cycling=True,video_list_map_linking=True,stable_numbers=True,
        filter_edits_correct_original_event=True,multi_select_visibility=True,hidden_view_saved=True,
        hidden_events_kept_in_csv_and_training=True,small_window=True,qa_only=True)
    (out/'interactive_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    assert not errors,errors
    print(json.dumps(report));w.dirty=False;w.close()


if __name__=='__main__':main()
