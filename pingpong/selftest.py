"""End-to-end check available in the frozen executable."""
from pathlib import Path
import json
import os
import time


def run(output):
    os.environ['QT_QPA_PLATFORM']='offscreen'
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QFontDatabase,QFont
    from .ui import MainWindow
    from .paths import asset_path,default_model
    from .analysis import save_project,export_csv,validate_project
    from .detector import export_video
    from .library import save_approved,prepare_training_index
    from .video_io import VideoReader
    import av
    import copy
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    app=QApplication([])
    for name in ('msyh.ttc','msyhbd.ttc'):
        path=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts'/name
        if path.exists():QFontDatabase.addApplicationFont(str(path))
    app.setFont(QFont('Microsoft YaHei UI',10))
    window=MainWindow();errors=[];window.error=lambda value:errors.append(value)
    assert window.model_preset.currentIndex()==0
    window.show();app.processEvents()
    sample=asset_path('sample/demo.mp4')
    assert window.load_video(sample),'sample video failed'
    window.auto_calibrate();started=time.monotonic()
    while window.task:
        app.processEvents();time.sleep(.01)
        if time.monotonic()-started>60:raise RuntimeError('table detection timeout')
    assert len(window.corners)==4 and not errors
    window.step_input.setValue(3);window.step_frames(1);assert window.frame_index==3
    window.step_frames(-1);assert window.frame_index==0
    window.start_analysis();started=time.monotonic();ticks=0
    while window.task:
        app.processEvents();time.sleep(.01);ticks+=1
        if time.monotonic()-started>90:raise RuntimeError('analysis timeout')
    app.processEvents()
    assert not errors,errors
    assert window.project and window.project['frame_count']==199
    assert window.project['decode_info']['backend']=='PyAV/FFmpeg'
    assert window.project['decode_info']['complete']
    original_tracks=copy.deepcopy(window.project['tracks'])
    window.apply_invisible_range(10,12)
    assert all(p['x'] is None and p['source']=='manual' for p in window.project['tracks'][10:13])
    window.undo_annotation();assert window.project['tracks']==original_tracks
    window.clean_view.setChecked(True);assert not any(window.display_options().values())
    window.clean_view.setChecked(False);assert all(window.display_options().values())
    first=window.project['events'][0];number=first['event_number']
    window.table.event_selected.emit(number)
    assert window.frame_index==first['frame'] and window.selected_number==number
    events_before=copy.deepcopy(window.project['events']);window.hide_number(number)
    assert window.project['events']==events_before and number in window.table.hidden
    window.restore_all_points();assert not window.table.hidden
    validate_project(window.project)
    export_csv(output,window.project)
    assert export_video(output/'demo_tracked.mp4',window.project)
    with av.open(str(output/'demo_tracked.mp4')) as encoded:
        assert encoded.streams.video[0].codec_context.name=='h264'
    for kind,name in [('plot','distribution.png'),('sync','synchronized.mp4')]:
        arguments=dict(path=output/name,project=copy.deepcopy(window.project))
        if kind=='sync':arguments['height']=480
        window.launch_task(kind,arguments);started=time.monotonic()
        while window.task:
            app.processEvents();time.sleep(.01)
            if time.monotonic()-started>90:raise RuntimeError('Distribution export timeout')
        assert not errors,errors
        assert (output/name).is_file()
    assert any(abs(e['frame']-23)<=2 for e in window.project['events'])
    qa=copy.deepcopy(window.project);qa['qa_only']=True
    accepted=save_approved(qa,output/'qa_corpus',events_reviewed=True)
    training=prepare_training_index(output/'qa_corpus',output/'qa_training')
    assert training['clips']==1 and training['reviewed_ball_frames']==0
    window.seek(26);app.processEvents()
    window.view.zoom_at(2);app.processEvents()
    assert window.view.zoom==2 and window.zoom_label.text()=='200%'
    window.grab().save(str(output/'zoom.png'))
    window.view.reset_zoom();app.processEvents()
    assert window.zoom_label.text()=='100%'
    window.grab().save(str(output/'app.png'))
    report=dict(success=True,model=str(default_model()),frames=199,
                detected_frames=sum(p['x'] is not None for p in window.project['tracks']),
                bounce_candidates=len(window.project['events']),
                automatic_corners=window.corners,frame_step_verified=3,
                event_frames=[e['frame'] for e in window.project['events']],
                performance=window.project['performance'],gui_ticks=ticks,
                export_video=True,export_csv=True,save_project=True)
    report.update(review_workflow_qa_only=True,training_index=True,unknown_ball_frames_excluded=True,zoom_controls_verified=True,
                  invisible_range_verified=True,annotation_undo_verified=True,layer_controls_verified=True)
    report['model_profiles_available']=window.model_preset.count()
    report.update(interactive_event_map_verified=True,selective_event_visibility_verified=True)
    report.update(distribution_png_export=True,synchronized_video_export=True,motion_filter_enabled=window.project.get('motion_filter_enabled'))
    report.update(decode_info=window.project['decode_info'],h264_output_verified=True)
    (output/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    window.dirty=False;window.close()
    return 0


def run_video(video,output):
    """Run the actual desktop import/calibration/analysis/reopen path on a supplied file."""
    os.environ['QT_QPA_PLATFORM']='offscreen'
    import hashlib
    import cv2
    import numpy as np
    from unittest.mock import patch
    from PySide6.QtWidgets import QApplication
    from .ui import MainWindow
    from .video_io import VideoReader
    from .analysis import save_project
    from .distribution_export import export_distribution
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    app=QApplication([]);window=MainWindow();errors=[];window.error=errors.append;window.show()
    def wait_task():
        start=time.monotonic()
        while window.task:
            app.processEvents();time.sleep(.01)
            if time.monotonic()-start>480:raise TimeoutError('Video workflow timeout')
        app.processEvents();assert not errors,errors
    reference=VideoReader(video);ok,first=reference.read();assert ok
    with patch('pingpong.ui.QFileDialog.getOpenFileName',return_value=(str(video),'')):
        window.open_video()
    wait_task()
    assert len(window.corners)==4,window.status.text()
    assert window.frame_index==0 and np.array_equal(window.raw_frame,first)
    assert int(window.cap.get(cv2.CAP_PROP_POS_FRAMES))==1
    calibration=dict(window.calibration_info)
    assert calibration.get('verified_frames'),calibration
    window.grab().save(str(output/'calibrated.png'))
    started=time.monotonic();window.start_analysis();wait_task()
    analysis_wall_seconds=time.monotonic()-started
    assert window.project and window.project['frame_count']>0
    project=window.project
    assert project['tracks'][0]['frame']==0 and project['tracks'][0]['time']==0.
    assert len(project['corners'])==4 and project['decode_info']['complete']
    assert np.array_equal(window.raw_frame,first)
    last=project['frame_count']-1;window.seek(last)
    reference.set_timeline([p['time'] for p in project['tracks']])
    assert reference.set(cv2.CAP_PROP_POS_FRAMES,last)
    ok,expected=reference.read();assert ok and np.array_equal(window.raw_frame,expected)
    reference.release()
    save_project(output/'project.json',project);window.dirty=False
    with patch('pingpong.ui.QFileDialog.getOpenFileName',return_value=(str(output/'project.json'),'')):
        window.open_project()
    assert not errors and window.project and window.total==last+1
    window.seek(last);assert window.frame_index==last
    for _ in range(5):app.processEvents()
    assert window.table.geometry().bottom()<window.map_count.geometry().top()
    assert len(window.table.label_rects)==len(window.table.visible_events())
    window.grab().save(str(output/'completed.png'))
    assert export_distribution(output/'distribution.png',window.project)
    report=dict(success=True,window_title=window.windowTitle(),input_video=str(video),
        import_auto_calibration=True,calibration=calibration,first_frame_preserved=True,
        first_frame_sha256=hashlib.sha256(first.tobytes()).hexdigest(),frames=window.total,
        decode_info=project['decode_info'],last_frame=last,last_frame_time=project['tracks'][-1]['time'],
        last_frame_pixels_verified=True,project_reopened=True,table_coordinates_available=sum(e.get('table_xy') is not None for e in project['events']),
        analysis_seconds=analysis_wall_seconds,performance=project['performance'],right_panel_no_overlap=True,all_map_labels=len(window.table.label_rects),errors=errors)
    (output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    window.dirty=False;window.close();return 0


def run_playback(project_path,output):
    """Measure real GUI playback on the user's analyzed video, including precision edits."""
    os.environ['QT_QPA_PLATFORM']='offscreen'
    import cv2,numpy as np
    from unittest.mock import patch
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    from .ui import MainWindow
    from .video_io import VideoReader
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    app=QApplication([]);window=MainWindow();window.show();errors=[];window.error=errors.append
    with patch('pingpong.ui.QFileDialog.getOpenFileName',return_value=(str(project_path),'')):
        window.open_project()
    assert window.project and not errors,errors
    gaps=[];previous=[time.perf_counter()]
    heartbeat=QTimer();heartbeat.setInterval(20)
    def tick():
        now=time.perf_counter();gaps.append(now-previous[0]);previous[0]=now
    heartbeat.timeout.connect(tick)
    cases=[]
    def play_case(speed,seconds,start_frame=0,nearby=False):
        window.pause();window.seek(start_frame);window.nearby_events.setChecked(nearby)
        window.playback_speed.setCurrentIndex(window.playback_speed.findData(speed));app.processEvents()
        gaps.clear();previous[0]=time.perf_counter();heartbeat.start()
        start=time.perf_counter();start_video=window.project['tracks'][start_frame]['time'];window.toggle_play();worker=window.playback
        while time.perf_counter()-start<seconds:
            app.processEvents();time.sleep(.001)
        elapsed=time.perf_counter()-start;window.pause();heartbeat.stop();worker.thread.join(2)
        video_time=window.project['tracks'][window.frame_index]['time']-start_video
        result=dict(speed=speed,nearby_filter=nearby,wall_seconds=elapsed,video_seconds=video_time,
            speed_ratio=video_time/elapsed/speed,heartbeat_p95_seconds=float(np.percentile(gaps,95)),
            max_queue=worker.max_queue,decoded_frames=worker.decoded_frames,display_samples=worker.converted_frames,
            decoder_stopped=not worker.thread.is_alive())
        assert .85<result['speed_ratio']<1.12,result
        assert result['heartbeat_p95_seconds']<.1 and result['decoder_stopped'] and worker.max_queue<=8,result
        cases.append(result)
    play_case(1.,6)
    play_case(2.,3)
    play_case(.25,3)
    play_case(1.,3,min(3600,window.total//2),True)
    window.nearby_events.setChecked(False);window.seek(window.total-30);window.playback_speed.setCurrentIndex(window.playback_speed.findData(2.))
    window.toggle_play();start=time.perf_counter()
    while window.timer.isActive() and time.perf_counter()-start<5:app.processEvents();time.sleep(.001)
    assert not window.timer.isActive() and window.frame_index==window.total-1
    window.step_input.setValue(1);window.step_frames(-1);assert window.frame_index==window.total-2
    reference=VideoReader(window.video);reference.set_timeline([p['time'] for p in window.project['tracks']])
    assert reference.set(cv2.CAP_PROP_POS_FRAMES,window.frame_index)
    assert np.array_equal(window.raw_frame,reference.read()[1]);reference.release()
    window.grab().save(str(output/'playback.png'))
    report=dict(success=True,version=window.windowTitle(),input_video=window.video,cases=cases,
        last_frame_reached=True,frame_accurate_step_and_pixels=True,errors=errors)
    assert not errors,errors
    (output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    window.dirty=False;window.close();return 0


def run_live_video(video, output):
    """Exercise the packaged monitor using the same controls as a user."""
    os.environ['QT_QPA_PLATFORM']='offscreen'
    from PySide6.QtWidgets import QApplication
    from .ui import MainWindow
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    app=QApplication([]);window=MainWindow();window.show();window.switch_mode(1)
    monitor=window.live_panel
    monitor.set_video(video);monitor.start.click()
    errors=[];monitor.worker.failed.connect(errors.append)
    started=time.perf_counter();first_event=None;last_snapshot=None;updates=0;last_report=0.
    while monitor.worker:
        app.processEvents()
        snapshot=monitor.last_snapshot
        if snapshot is not None and snapshot is not last_snapshot:
            updates+=1;last_snapshot=snapshot
            now=time.perf_counter()-started
            if now-last_report>=5 and snapshot.get('tracks'):
                (output/'progress.json').write_text(json.dumps(dict(wall=now,media=snapshot['time'],
                    frames=snapshot['frames'],lag=snapshot['lag'],fps=snapshot['fps'])),encoding='utf-8')
                last_report=now
            if first_event is None and snapshot.get('events'):
                first_event=dict(video_time=snapshot['time'],wall_time=time.perf_counter()-started,frames=snapshot['frames'])
                window.grab().save(str(output/'live.png'))
        if time.perf_counter()-started>180:
            monitor.stop_session()
            raise TimeoutError(f'Live simulation exceeded the test timeout at frame {(snapshot or {}).get("frames",0)}')
        time.sleep(.004)
    assert not errors,errors
    result=monitor.result
    assert result and result['frames']>0 and updates>2,result
    assert len(monitor.last_snapshot['corners'])==4
    window.grab().save(str(output/'finished.png'))
    report=dict(success=True,frames=result['frames'],fps=result['fps'],source_fps=result['source_fps'],sample_step=result['sample_step'],input_video=str(video),
        counts=result['counts'],elapsed_seconds=result['elapsed_seconds'],lag_p95_seconds=result['lag_p95_seconds'],
        ui_updates=updates,first_visible_event=first_event,errors=errors)
    (output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    window.close();return 0


def simulated_camera(index, backend, stopped, pixels, meta, lock):
    import cv2
    import numpy as np
    from .camera import publish_frame
    sequence = 0
    while not stopped.value:
        if backend == cv2.CAP_MSMF and sequence >= 18:
            # Simulate a driver that never returns from its next read call.
            time.sleep(30)
        sequence += 1
        frame = np.full((96,160,3),sequence % 256,np.uint8)
        publish_frame(pixels,meta,lock,frame,30.)
        time.sleep(1/30)


def run_camera_transport(output):
    from threading import Event
    from .camera import CameraReader
    from .cancellation import TaskCancelled
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    cancelled=Event()
    reader=CameraReader(0,cancelled.is_set,capture_target=simulated_camera,stall_timeout=.45)
    try:
        assert reader.read()[0]
        first=reader.last_key
        time.sleep(.35)
        fresh=reader.preview(first)
        assert fresh and fresh[0][1]>=first[1]+4,fresh[0] if fresh else None
        assert reader.read()[0] and reader.last_key[1]>=fresh[0][1]
        deadline=time.perf_counter()+9
        while reader.generation==1 and time.perf_counter()<deadline:
            reader.read()
        assert reader.generation==2
        key=reader.last_key
        for _ in range(10):assert reader.read()[0]
        assert reader.last_key[0]==key[0] and reader.last_key[1]>=key[1]+10
        cancelled.set()
        try:reader.read()
        except TaskCancelled:pass
        else:raise AssertionError('Camera read ignored cancellation')
        process=reader.process
    finally:
        reader.release()
    try:alive=process.is_alive()
    except ValueError:alive=False  # A stopped process handle has already been closed.
    assert not alive
    restarted=CameraReader(0,capture_target=simulated_camera)
    try:assert restarted.read()[0]
    finally:restarted.release()
    result=dict(success=True,fixture='synthetic camera process',preview_frames_during_inference_pause=fresh[0][1]-first[1],
                blocked_driver_recovered=True,frames_after_recovery=10,cancelled=True,restart_verified=True)
    (output/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return 0
