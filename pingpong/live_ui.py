"""Live monitor. A one-frame mailbox keeps rendering independent of inference."""
from pathlib import Path
from threading import Event
import time

import cv2
import numpy as np
from PySide6.QtCore import QThread, QTimer, Signal, QUrl, Qt
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QWidget, QFrame, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QSpinBox, QFileDialog, QListWidget)

from .analysis import homography
from .cancellation import TaskCancelled
from .detector import render_frame
from .events import event_kind
from .live import run_live
from .paths import asset_path
from .point_map import TableView


class LiveWorker(QThread):
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, source, settings, corners):
        super().__init__()
        self.source = source
        self.settings = settings
        self.corners = corners
        self.cancel = Event()
        self.snapshot = None
        self.camera = None

    def publish(self, snapshot):
        self.snapshot = snapshot

    def attach_camera(self, reader):
        self.camera = reader

    def run(self):
        try:
            self.completed.emit(run_live(self.source,*self.settings,self.cancel.is_set,self.publish,self.corners,
                                         on_camera=self.attach_camera))
        except TaskCancelled:
            self.completed.emit(None)
        except Exception as error:
            self.failed.emit(str(error))


def card():
    frame = QFrame()
    frame.setObjectName('panel')
    return frame


class LivePanel(QWidget):
    stopped = Signal()

    def __init__(self, view_type, settings):
        super().__init__()
        self.settings = settings
        self.video = None
        self.worker = None
        self.last_snapshot = None
        self.preview_key = None
        self.last_frame = None
        self.manual_corners = []
        self.marking = False
        self.last_event = 0
        self.result = None
        root = QVBoxLayout(self)
        root.setContentsMargins(0,0,0,0)
        root.setSpacing(14)
        controls = QHBoxLayout()
        self.mode = QComboBox()
        self.mode.addItem('视频模拟','video')
        self.mode.addItem('Windows 摄像头','camera')
        self.mode.currentIndexChanged.connect(self.source_changed)
        controls.addWidget(self.mode)
        self.choose = self.button(controls,'选择视频',self.choose_video)
        self.demo = self.button(controls,'内置示例',self.choose_demo)
        self.device = QSpinBox()
        self.device.setRange(0,15)
        self.device.setPrefix('设备 ')
        self.device.valueChanged.connect(self.reset_calibration)
        controls.addWidget(self.device)
        self.pair = self.button(controls,'蓝牙配对',lambda:QDesktopServices.openUrl(QUrl('ms-settings:bluetooth')))
        self.camera_settings = self.button(controls,'系统摄像头',lambda:QDesktopServices.openUrl(QUrl('ms-settings:camera')))
        controls.addStretch()
        self.mark = self.button(controls,'手动标台',self.mark_table)
        self.start = self.button(controls,'开始监测',self.start_session,'primary')
        self.stop = self.button(controls,'停止',self.stop_session)
        root.addLayout(controls)

        body = QHBoxLayout()
        body.setSpacing(14)
        left = card()
        video_layout = QVBoxLayout(left)
        video_layout.setContentsMargins(16,14,16,14)
        caption = QHBoxLayout()
        title = QLabel('固定机位画面')
        title.setObjectName('section')
        caption.addWidget(title)
        caption.addStretch()
        self.clock = QLabel('00:00.000')
        self.clock.setObjectName('muted')
        caption.addWidget(self.clock)
        video_layout.addLayout(caption)
        self.view = view_type()
        self.view.clicked.connect(self.corner_clicked)
        video_layout.addWidget(self.view,1)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setObjectName('muted')
        video_layout.addWidget(self.status)
        body.addWidget(left,1)

        side = card()
        side.setFixedWidth(318)
        right = QVBoxLayout(side)
        right.setContentsMargins(14,14,14,14)
        title = QLabel('实时事件')
        title.setObjectName('section')
        right.addWidget(title)
        metrics = QHBoxLayout()
        self.metrics = {}
        for key, name in [('bounce','触台'),('net','触网'),('out','出界')]:
            label = QLabel(f'0\n{name}')
            label.setAlignment(Qt.AlignCenter)
            label.setObjectName('metric')
            label.setMinimumHeight(61)
            metrics.addWidget(label,1)
            self.metrics[key] = (label,name)
        right.addLayout(metrics)
        self.map = TableView()
        self.map.numbers = False
        self.map.setMinimumHeight(230)
        right.addWidget(self.map,1)
        self.log = QListWidget()
        self.log.setMaximumHeight(130)
        self.log.setAccessibleName('最近候选事件')
        right.addWidget(self.log)
        note = QLabel('候选事件会在观察到后续轨迹后显示。')
        note.setWordWrap(True)
        note.setObjectName('muted')
        right.addWidget(note)
        body.addWidget(side)
        root.addLayout(body,1)
        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self.refresh)
        self.source_changed()
        self.update_controls()

    def button(self, layout, title, callback, style=''):
        button = QPushButton(title)
        if style:button.setObjectName(style)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    def update_controls(self):
        busy = self.worker is not None
        for widget in (self.mode,self.choose,self.demo,self.device,self.pair,self.camera_settings):
            widget.setEnabled(not busy)
        self.start.setEnabled(not busy and (self.mode.currentData()=='camera' or self.video is not None) and not self.marking)
        self.stop.setEnabled(busy and not self.worker.cancel.is_set())
        self.mark.setEnabled(not busy and self.last_frame is not None)

    def reset_calibration(self,*_):
        self.manual_corners = []
        self.last_frame = None
        self.marking = False
        self.view.image = None
        self.view.update()
        self.update_controls()

    def source_changed(self,*_):
        camera = self.mode.currentData()=='camera'
        for widget in (self.device,self.pair,self.camera_settings):widget.setVisible(camera)
        for widget in (self.choose,self.demo):widget.setVisible(not camera)
        self.reset_calibration()
        if camera:
            self.status.setText('先在 Windows 配对设备，确认系统能预览摄像头，再选择设备编号。')
        else:
            self.status.setText(Path(self.video).name if self.video else '选择固定机位视频，或载入内置示例。')

    def set_video(self, path):
        self.video = str(path)
        self.mode.setCurrentIndex(0)
        self.source_changed()

    def choose_video(self):
        path,_ = QFileDialog.getOpenFileName(self,'选择实时模拟视频','','视频 (*.mp4 *.mov *.avi *.mkv *.webm);;所有文件 (*)')
        if path:self.set_video(path)

    def choose_demo(self):
        self.set_video(asset_path('sample/demo.mp4'))

    def start_session(self):
        if self.worker or self.marking:return
        source = self.device.value() if self.mode.currentData()=='camera' else self.video
        if source is None:return
        self.result = None
        self.last_snapshot = None
        self.preview_key = None
        self.last_event = 0
        self.log.clear()
        self.map.events = []
        self.map.update()
        for label, name in self.metrics.values():label.setText(f'0\n{name}')
        self.clock.setText('00:00.000')
        self.status.setText('正在打开输入并标定球台…')
        self.worker = LiveWorker(source,self.settings(),list(self.manual_corners))
        self.worker.completed.connect(self.completed)
        self.worker.failed.connect(self.failed)
        self.worker.finished.connect(self.finished)
        self.update_controls()
        self.timer.start()
        self.worker.start()

    def stop_session(self):
        if self.worker:
            self.worker.cancel.set()
            self.status.setText('正在停止监测…')
            self.update_controls()

    def refresh(self):
        snapshot = self.worker.snapshot if self.worker else None
        camera = self.worker.camera if self.worker else None
        preview_only = False
        if camera and not self.worker.cancel.is_set():
            now = time.perf_counter()
            age = now - (snapshot or {}).get('captured_at', 0.)
            if age > .25:
                preview_only = True
                preview = camera.preview(self.preview_key)
                if preview is not None and now - preview[2] < .6:
                    self.preview_key, self.last_frame, _, _ = preview
                    self.view.set_frame(self.last_frame)
                    status = (snapshot or {}).get('status', '正在准备检测…')
                    self.status.setText(status + ' · 摄像头预览持续采集')
                elif age > .75:
                    self.status.setText(camera.status if camera.channel is None or '重连' in camera.status
                                        else '等待摄像头新画面…')
        if snapshot is None or snapshot is self.last_snapshot:return
        self.last_snapshot = snapshot
        if self.last_frame is None:preview_only = False
        frame = snapshot['frame']
        tracks = snapshot.get('tracks',[])
        if tracks:
            if not preview_only:
                if frame.shape[1::-1]!=tuple(snapshot['size']):
                    frame=cv2.resize(frame,tuple(snapshot['size']))
                base = tracks[0]['frame']
                event_list = [dict(e,frame=e['frame']-base) for e in snapshot['events'] if e['frame']>=base]
                image = render_frame(frame,dict(tracks=tracks,events=event_list,corners=snapshot['corners'],fps=snapshot['fps']),len(tracks)-1)
            timestamp = snapshot['time']
            self.clock.setText(f'{int(timestamp//60):02d}:{timestamp%60:06.3f}')
            new = [e for e in snapshot['events'] if e['event_number']>self.last_event]
            if new:
                self.last_event = new[-1]['event_number']
                self.map.events = snapshot['events']
                if event_kind(new[-1])=='bounce':self.map.selected=new[-1]['event_number']
                self.map.update()
                for key,(label,name) in self.metrics.items():label.setText(f'{snapshot["counts"][key]}\n{name}')
                names = {'bounce':'触台','net':'疑似触网','out':'疑似出界'}
                for event in new:
                    self.log.insertItem(0,f'{event["time"]:6.2f}s  ·  {names[event_kind(event)]}')
                while self.log.count()>20:self.log.takeItem(self.log.count()-1)
            suffix = f' · 检测 {snapshot["fps"]:.0f} FPS · 延迟 {snapshot["lag"]*1000:.0f} ms'
            if snapshot['sample_step']>1:suffix+=f' · 每 {snapshot["sample_step"]} 帧取 1 帧'
        else:
            image = frame
            suffix = ''
        if not preview_only:
            self.last_frame = frame
            self.view.set_frame(image)
            if not self.worker.cancel.is_set():self.status.setText(snapshot['status']+suffix)

    def completed(self, result):
        self.refresh()
        self.result = result
        self.status.setText(f'监测完成 · 处理 {result["frames"]} 帧' if result else '监测已停止')

    def failed(self, message):
        self.refresh()
        self.status.setText(message)

    def finished(self):
        self.timer.stop()
        self.worker.deleteLater()
        self.worker = None
        self.update_controls()
        self.stopped.emit()

    def mark_table(self):
        self.marking = True
        self.manual_corners = []
        self.view.annotation_mode = True
        self.status.setText('沿台面边缘依次点击远左、远右、近右、近左四个角。')
        self.view.set_frame(self.last_frame)
        self.update_controls()

    def corner_clicked(self,x,y):
        if not self.marking:return
        self.manual_corners.append([x,y])
        if len(self.manual_corners)==4:
            try:homography(self.manual_corners)
            except ValueError as error:
                self.status.setText(str(error));self.manual_corners=[]
            else:
                self.marking=False
                self.view.annotation_mode=False
                self.status.setText('手动标台完成，可以开始监测。')
        image = self.last_frame.copy()
        for px,py in self.manual_corners:
            cv2.circle(image,(round(px),round(py)),8,(85,238,160),-1)
        if len(self.manual_corners)>1:
            cv2.polylines(image,[np.array(self.manual_corners,np.int32)],len(self.manual_corners)==4,(85,238,160),2)
        self.view.set_frame(image)
        self.update_controls()
