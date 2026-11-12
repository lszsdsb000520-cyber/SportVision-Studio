"""Standalone desktop review UI; all analysis runs off the GUI thread."""
from pathlib import Path
import json
import math
import sys
import threading
import time
import traceback
import copy
import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QRectF, QPointF,QUrl,QItemSelectionModel
from PySide6.QtGui import QColor, QFont, QFontDatabase, QImage, QPainter, QPen, QPolygonF,QPalette,QShortcut,QKeySequence,QDesktopServices
from PySide6.QtWidgets import (QApplication,QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,
    QLabel,QPushButton,QFileDialog,QSlider,QProgressBar,QDoubleSpinBox,QTableWidget,
    QTableWidgetItem,QHeaderView,QMessageBox,QCheckBox,QSplitter,QFrame,QComboBox,
    QScrollArea,QSizePolicy,QSpinBox,QListWidget,QDialog,QLineEdit,QMenu,QLayout,QStackedWidget)
from .paths import app_root, asset_path, default_model,model_threshold
from .analysis import (TrackPoint,homography,map_point,propose_bounces,export_csv,
                       save_project,validate_project)
from .detector import analyze_video,render_frame,export_video
from .boundary import TableBoundaryDetector
from .events import analyze_events,event_kind,table_half,rally_hints,ALGORITHM
from .library import save_approved,prepare_training_index,approval_root,log_edit,invalidate_review,digest
from .media import resolve_video,workspace_root,media_details,cleanup_legacy_aliases
from .review_dialogs import EventDialog,ApprovalDialog,InvisibleRangeDialog,DistributionExportDialog
from .distribution_export import export_distribution,export_synchronized
from .annotations import AnnotationHistory,mark_invisible_range
from .point_map import TableView
from .playback import PlaybackDecoder
from .review_state import ensure_numbers,hidden_numbers,matches_filter,reuse_numbers
from .video_io import VideoReader
from .live_ui import LivePanel


def configure_app(app):
    if app.property('pingpong_theme_ready'):return
    import os
    # Load installed Chinese fonts explicitly; native and offscreen rendering use the same font.
    for name in ('msyh.ttc','msyhbd.ttc'):
        path=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts'/name
        if path.exists():QFontDatabase.addApplicationFont(str(path))
    app.setStyle('Fusion')
    palette=QPalette()
    for role,color in [(QPalette.Window,'#0b1424'),(QPalette.WindowText,'#e7eef8'),
        (QPalette.Base,'#132239'),(QPalette.AlternateBase,'#192a40'),(QPalette.Text,'#e7eef8'),
        (QPalette.Button,'#1b2b43'),(QPalette.ButtonText,'#e7eef8'),
        (QPalette.Highlight,'#286478'),(QPalette.HighlightedText,'#ffffff'),
        (QPalette.ToolTipBase,'#23364f'),(QPalette.ToolTipText,'#e7eef8')]:palette.setColor(role,QColor(color))
    palette.setColor(QPalette.Disabled,QPalette.Text,QColor('#8191a9'))
    palette.setColor(QPalette.Disabled,QPalette.ButtonText,QColor('#8191a9'))
    app.setPalette(palette)
    font=QFont('Microsoft YaHei UI',10);font.setStyleStrategy(QFont.PreferAntialias)
    app.setFont(font);app.setProperty('pingpong_theme_ready',True)


class VideoView(QWidget):
    clicked=Signal(float,float)
    corner_moved=Signal(int,float,float,bool)
    zoom_changed=Signal(float)

    def __init__(self):
        super().__init__()
        self.image=None;self.corners=[];self.allow_drag=False;self.drag_index=None
        self.zoom=1.;self.pan=QPointF();self.pan_start=None
        self.pan_mode=False;self.annotation_mode=False;self.pending_click=None;self.press_position=None
        self.setMinimumSize(300,120)
        self.setCursor(Qt.CrossCursor)
        self.setToolTip('滚轮缩放；按住鼠标右键或中键拖动画面；左键标点或拖动球台四角。')

    def set_frame(self,frame):
        self.image=QImage(frame.data,frame.shape[1],frame.shape[0],frame.strides[0],QImage.Format_BGR888).copy()
        self.update()

    def image_rect(self):
        if self.image is None:return QRectF()
        w,h=self.image.width(),self.image.height()
        scale=min(self.width()/w,self.height()/h)*self.zoom
        dx=max(0,(w*scale-self.width())/2);dy=max(0,(h*scale-self.height())/2)
        self.pan=QPointF(float(np.clip(self.pan.x(),-dx,dx)),float(np.clip(self.pan.y(),-dy,dy)))
        return QRectF((self.width()-w*scale)/2+self.pan.x(),(self.height()-h*scale)/2+self.pan.y(),w*scale,h*scale)

    def reset_zoom(self):
        self.zoom=1.;self.pan=QPointF();self.pan_start=None
        self.zoom_changed.emit(self.zoom);self.update()

    def set_pan_mode(self,enabled):
        self.pan_mode=enabled;self.pending_click=None;self.press_position=None
        self.setCursor(Qt.OpenHandCursor if enabled else Qt.CrossCursor)

    def zoom_at(self,factor,anchor=None):
        if self.image is None:return
        anchor=anchor if anchor is not None else QPointF(self.rect().center())
        old=self.image_rect();new_zoom=float(np.clip(self.zoom*factor,1.,8.))
        ratio=new_zoom/self.zoom
        left=anchor.x()-(anchor.x()-old.left())*ratio
        top=anchor.y()-(anchor.y()-old.top())*ratio
        self.pan=QPointF(left-(self.width()-old.width()*ratio)/2,top-(self.height()-old.height()*ratio)/2)
        self.zoom=new_zoom;self.image_rect()
        self.zoom_changed.emit(self.zoom);self.update()

    def wheelEvent(self,event):
        if self.image is None:event.ignore();return
        self.zoom_at(1.2**(event.angleDelta().y()/120),event.position());event.accept()

    def paintEvent(self,event):
        p=QPainter(self);p.fillRect(self.rect(),QColor('#080f1d'))
        if self.image is not None:p.drawImage(self.image_rect(),self.image)
        else:
            p.setPen(QColor('#8d9bb3'));p.setFont(QFont('Microsoft YaHei UI',13))
            p.drawText(self.rect(),Qt.AlignCenter,'导入一段视频，开始追踪每一次来回')

    def mousePressEvent(self,event):
        if self.image is None:return
        self.pending_click=None;self.press_position=None
        if event.button() in (Qt.RightButton,Qt.MiddleButton) or (event.button()==Qt.LeftButton and self.pan_mode):
            self.pan_start=event.position();self.setCursor(Qt.ClosedHandCursor);return
        if event.button()!=Qt.LeftButton:return
        r=self.image_rect();point=event.position()
        if r.contains(point):
            x=(point.x()-r.x())/r.width()*self.image.width();y=(point.y()-r.y())/r.height()*self.image.height()
            if self.allow_drag and len(self.corners)==4:
                distances=[np.hypot(x-cx,y-cy)*r.width()/self.image.width() for cx,cy in self.corners]
                if min(distances)<=16:self.drag_index=int(np.argmin(distances));return
            self.pending_click=(x,y);self.press_position=point

    def mouseMoveEvent(self,event):
        if self.pan_start is not None:
            self.pan+=event.position()-self.pan_start;self.pan_start=event.position()
            self.image_rect();self.update();return
        if self.press_position is not None and not self.annotation_mode and self.zoom>1 and (event.position()-self.press_position).manhattanLength()>4:
            self.pan+=event.position()-self.press_position;self.pan_start=event.position()
            self.pending_click=None;self.press_position=None;self.setCursor(Qt.ClosedHandCursor);self.image_rect();self.update();return
        if self.drag_index is not None:self.move_corner(event,False)

    def mouseReleaseEvent(self,event):
        if self.pan_start is not None:
            self.pan_start=None;self.pending_click=None;self.press_position=None
            self.setCursor(Qt.OpenHandCursor if self.pan_mode else Qt.CrossCursor);return
        if self.drag_index is not None:self.move_corner(event,True);self.drag_index=None
        elif event.button()==Qt.LeftButton and self.pending_click is not None:self.clicked.emit(*self.pending_click)
        self.pending_click=None;self.press_position=None

    def move_corner(self,event,finished):
        r=self.image_rect();p=event.position()
        x=np.clip((p.x()-r.x())/r.width()*self.image.width(),0,self.image.width()-1)
        y=np.clip((p.y()-r.y())/r.height()*self.image.height(),0,self.image.height()-1)
        self.corner_moved.emit(self.drag_index,float(x),float(y),finished)


class Task(QThread):
    progress=Signal(int,int,object,object)
    result=Signal(object)
    failed=Signal(str)

    def __init__(self,kind,args):
        super().__init__();self.kind=kind;self.args=args;self.cancel=threading.Event();self.last=0

    def update_progress(self,done,total,frame,point):
        now=time.monotonic()
        if now-self.last>.2 or done==total:
            self.progress.emit(done,total,frame,point);self.last=now

    def run(self):
        try:
            if self.cancel.is_set():self.result.emit(None);return
            if self.kind=='analyze':
                value=analyze_video(**self.args,progress=self.update_progress,cancelled=self.cancel.is_set)
            elif self.kind=='calibrate':
                value=TableBoundaryDetector(acceleration=self.args['acceleration']).detect_clip(
                    self.args['frame'],video=self.args.get('video'),frame_index=self.args.get('frame_index',0),
                    fps=self.args.get('fps',30.),cancelled=self.cancel.is_set,fixed_camera=True)
                value['task_type']='calibration'
                if self.cancel.is_set():value=None
            elif self.kind=='approve':
                value=save_approved(**self.args,cancelled=self.cancel.is_set());value['task_type']='approve'
            elif self.kind=='training':
                value=prepare_training_index(cancelled=self.cancel.is_set);value['task_type']='training'
            elif self.kind in ('plot','sync'):
                path=export_distribution(**self.args,cancelled=self.cancel.is_set) if self.kind=='plot' else export_synchronized(**self.args,progress=self.update_progress,cancelled=self.cancel.is_set)
                value=dict(task_type='artifact',path=path) if path else None
            else:
                value=export_video(**self.args,progress=self.update_progress,cancelled=self.cancel.is_set)
            if self.cancel.is_set() and self.kind in ('analyze','calibrate'):value=None
            self.result.emit(value)
        except InterruptedError:self.result.emit(None)
        except ValueError as error:
            if self.kind=='calibrate':self.result.emit(dict(task_type='calibration_unavailable',message=str(error)))
            else:self.failed.emit(traceback.format_exc())
        except Exception:
            self.failed.emit(traceback.format_exc())


STYLE='''
QMainWindow,QWidget { background:#0b1424; color:#e7eef8; font-family:"Microsoft YaHei UI"; font-size:13px; }
QLabel { background:transparent; }
QFrame#panel { background:#111d30; border:1px solid #22334d; border-radius:12px; }
QLabel#muted { color:#90a2ba; background:transparent; }
QLabel#title { font-size:24px; font-weight:700; background:transparent; }
QLabel#section { font-size:15px; font-weight:600; color:#e3edf7; background:transparent; }
QPushButton { background:#1b2b43; border:1px solid #30435f; padding:10px 12px; border-radius:7px; text-align:left; }
QPushButton:hover { background:#263d59; border-color:#4a718d; }
QPushButton:checked { background:#236477; border-color:#5bd2c2; }
QPushButton:disabled { color:#55647a; background:#142034; border-color:#233049; }
QPushButton#primary { background:#33c1b0; color:#081f28; border:none; font-weight:700; }
QPushButton#primary:hover { background:#61d9c5; }
QPushButton#danger { color:#ffc49a; }
QDoubleSpinBox { background:#17283e; border:1px solid #30435f; border-radius:6px; padding:7px; }
QSpinBox,QDoubleSpinBox,QComboBox { background:#17283e; color:#edf5ff; border:1px solid #3c536c; border-radius:6px; min-height:28px; padding:3px 8px; selection-background-color:#286478; selection-color:white; }
QComboBox { padding-right:26px; }
QComboBox::drop-down { width:22px; border-left:1px solid #3c536c; }
QComboBox::down-arrow { image:url("__DOWN__"); width:14px; height:14px; }
QSpinBox::up-button,QDoubleSpinBox::up-button { subcontrol-origin:border; subcontrol-position:top right; width:20px; background:#29405a; border-top-right-radius:5px; }
QSpinBox::down-button,QDoubleSpinBox::down-button { subcontrol-origin:border; subcontrol-position:bottom right; width:20px; background:#29405a; border-bottom-right-radius:5px; }
QSpinBox::up-arrow,QDoubleSpinBox::up-arrow { image:url("__UP__"); width:11px; height:11px; }
QSpinBox::down-arrow,QDoubleSpinBox::down-arrow { image:url("__DOWN__"); width:11px; height:11px; }
QComboBox QAbstractItemView { background:#17283e; color:#edf5ff; selection-background-color:#286478; selection-color:white; border:1px solid #638095; min-height:110px; }
QScrollArea { background:transparent; border:0; }
QScrollBar:vertical { background:#101c2c; width:10px; margin:0; }
QScrollBar::handle:vertical { background:#405a70; min-height:30px; border-radius:4px; }
QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical { height:0; }
QProgressBar { background:#15233a; border:none; border-radius:5px; height:10px; text-align:center; }
QProgressBar::chunk { background:#33c1b0; border-radius:5px; }
QSlider::groove:horizontal { background:#293b54; height:5px; border-radius:2px; }
QSlider::handle:horizontal { background:#6cddca; width:13px; margin:-5px 0; border-radius:6px; }
QTableWidget { background:#111d30; alternate-background-color:#162338; gridline-color:#22334b; border:0; selection-background-color:#264b60; }
QTableWidget::item { padding:5px; }
QTableWidget::item:selected { background:#286478; color:white; }
QHeaderView::section { background:#17273d; color:#a9bdd3; border:0; padding:8px; }
QCheckBox { background:transparent; spacing:8px; }
QToolTip { color:#e7eef8; background:#23364f; border:1px solid #52758c; }
QFrame#masthead { background:#e8eef5; border-radius:10px; }
QLabel#brand { color:#142640; font-size:16px; font-weight:800; }
QLabel#heroTitle { color:#111f33; font-size:29px; font-weight:800; }
QLabel#heroCaption { color:#4c6281; font-size:13px; }
QPushButton#nav { background:transparent; color:#314c6a; border:0; padding:8px 16px; font-weight:700; }
QPushButton#nav:checked { color:#fff; background:#2867c3; }
QPushButton#fold { background:transparent; border:1px solid #30435f; color:#a9c2df; padding:8px 10px; }
QPushButton#fold:checked { background:#1b2c44; color:white; }
QLabel#metric { background:#182944; color:#bde0ff; border:1px solid #304866; border-radius:8px; font-size:16px; font-weight:700; }
QPushButton#primary { background:#4b93ff; color:#09192d; }
QPushButton#primary:hover { background:#74adff; }
QPushButton#primary:disabled { background:#253851; color:#8195ae; }
QProgressBar::chunk { background:#4b93ff; }
QSlider::handle:horizontal { background:#77b0ff; }
'''


def panel():
    widget=QFrame();widget.setObjectName('panel');return widget


class CourtMark(QWidget):
    """Original table and ball line art for the masthead."""
    def __init__(self):
        super().__init__()
        self.setFixedSize(235,82)

    def paintEvent(self, _event):
        painter=QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        outline=QPolygonF([QPointF(46,58),QPointF(151,26),QPointF(211,42),QPointF(105,76)])
        painter.setPen(QPen(QColor('#6b98cc'),1.5))
        painter.setBrush(QColor('#cfdef0'))
        painter.drawPolygon(outline)
        painter.drawLine(QPointF(98,42),QPointF(158,59))
        painter.drawLine(QPointF(76,67),QPointF(181,34))
        painter.setPen(QPen(QColor('#3983df'),2))
        painter.drawLine(QPointF(29,47),QPointF(102,27))
        painter.drawLine(QPointF(60,24),QPointF(137,8))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor('#256ed0'))
        painter.drawEllipse(QPointF(154,11),6,6)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        configure_app(QApplication.instance())
        self.setWindowTitle('乒乓轨迹 Studio v8.1 · 比赛分析与实时监测')
        self.resize(1440,920);self.setMinimumSize(1120,740)
        self.cap=None;self.project=None;self.video=None;self.frame_index=0;self.raw_frame=None
        self.fps=30.;self.total=0;self.corners=[];self.calibrating=False;self.editing=False
        self.task=None;self.model_path=default_model();self.pending_close=False;self.dirty=False
        self.calibration_info={};self.corner_backup=None
        self.edit_event_ref=None;self.pending_add_event=False
        self.history=AnnotationHistory();self.range_start=None;self.range_end=None
        self.visible_rows=[];self.selected_number=None
        self.timer=QTimer(self);self.timer.timeout.connect(self.advance)
        self.timer.setTimerType(Qt.PreciseTimer);self.playback=None;self.play_wall=None;self.last_nearby_refresh=0.
        outer=QWidget();self.setCentralWidget(outer);layout=QVBoxLayout(outer);layout.setContentsMargins(22,18,22,14);layout.setSpacing(16)
        mast=QFrame();mast.setObjectName('masthead');mast_layout=QVBoxLayout(mast)
        mast_layout.setContentsMargins(20,10,20,10);mast_layout.setSpacing(2)
        header=QHBoxLayout();brand=QLabel('乒乓轨迹  /  STUDIO');brand.setObjectName('brand');header.addWidget(brand)
        header.addStretch()
        self.offline_tab=self.button(header,'离线分析',lambda:self.switch_mode(0),'nav')
        self.live_tab=self.button(header,'实时监测',lambda:self.switch_mode(1),'nav')
        for tab in (self.offline_tab,self.live_tab):tab.setCheckable(True)
        self.offline_tab.setChecked(True);mast_layout.addLayout(header)
        hero=QHBoxLayout();hero_text=QVBoxLayout();hero_text.setSpacing(3)
        self.hero_title=QLabel('比赛复盘');self.hero_title.setObjectName('heroTitle');hero_text.addWidget(self.hero_title)
        self.hero_caption=QLabel('导入固定机位视频，分析轨迹、复核落点并导出。');self.hero_caption.setObjectName('heroCaption');hero_text.addWidget(self.hero_caption)
        hero.addLayout(hero_text);hero.addStretch();hero.addWidget(CourtMark());mast_layout.addLayout(hero);layout.addWidget(mast)
        self.pages=QStackedWidget();layout.addWidget(self.pages,1)
        offline=QWidget();offline_layout=QVBoxLayout(offline);offline_layout.setContentsMargins(0,0,0,0)
        self.pages.addWidget(offline)
        body=QHBoxLayout();body.setSpacing(16);offline_layout.addLayout(body,1)
        side=panel();sl=QVBoxLayout(side);sl.setContentsMargins(14,16,14,16);sl.setSpacing(10)
        self.section(sl,'比赛素材')
        self.open_btn=self.button(sl,'＋  导入视频',self.open_video)
        self.project_btn=self.button(sl,'打开分析项目',self.open_project)
        self.filename=QLabel('尚未选择视频');self.filename.setWordWrap(True);self.filename.setObjectName('muted');sl.addWidget(self.filename)
        settings=self.fold(sl,'模型与参数')
        self.model_btn=self.button(settings,'选择检测模型',self.choose_model)
        self.model_label=QLabel(self.model_path.name if self.model_path.exists() else '模型未就绪');self.model_label.setObjectName('muted');settings.addWidget(self.model_label)
        self.model_preset=QComboBox();self.model_preset.setPlaceholderText('自定义模型')
        self.model_preset.addItem('通用场景（原模型）',str(default_model()))
        extended_path=default_model().parent/'detector_extended.onnx'
        if extended_path.exists():self.model_preset.addItem('侧视比赛（新数据微调）',str(extended_path))
        self.model_preset.setCurrentIndex(0)
        self.model_preset.setToolTip('新模型适合新增数据中的侧视比赛；旧场景误检可能增加。切换模型后需重新分析。')
        settings.addWidget(self.model_preset)
        label=QLabel('检测置信度阈值');label.setObjectName('muted');settings.addWidget(label)
        self.threshold=QDoubleSpinBox();self.threshold.setRange(.1,.95);self.threshold.setSingleStep(.05);self.threshold.setDecimals(2);self.threshold.setValue(model_threshold(self.model_path));settings.addWidget(self.threshold)
        self.acceleration=QComboBox();self.acceleration.addItem('自动选择高性能显卡','auto');self.acceleration.addItem('CPU 兼容模式','cpu');self.acceleration.addItem('显卡 0','0');self.acceleration.addItem('显卡 1','1');settings.addWidget(self.acceleration)
        self.motion_filter=QCheckBox('过滤持续静止杂点');self.motion_filter.setChecked(True)
        self.motion_filter.setToolTip('结合持续停留和局部画面变化过滤背景白点。');settings.addWidget(self.motion_filter)
        self.section(sl,'球台标定')
        self.auto_cal_btn=self.button(sl,'自动识别球台边界',self.auto_calibrate)
        self.auto_on_load=QCheckBox('导入时自动识别球台');self.auto_on_load.setChecked(True);settings.addWidget(self.auto_on_load)
        self.cal_btn=self.button(sl,'标定球台 · 选择四角',self.start_calibration)
        corners_settings=self.fold(sl,'角点操作')
        self.rotate_cal_btn=self.button(corners_settings,'调整球台方向（90°）',self.rotate_calibration)
        self.clear_cal_btn=self.button(corners_settings,'清除球台标定',self.clear_calibration)
        self.run_btn=self.button(sl,'开始分析',self.start_analysis,'primary')
        self.cancel_btn=self.button(sl,'取消当前任务',self.cancel_task,'danger');self.cancel_btn.setEnabled(False)
        self.section(sl,'输出与保存')
        self.save_btn=self.button(sl,'保存分析项目',self.save)
        outputs=self.fold(sl,'更多导出与资料')
        self.csv_btn=self.button(outputs,'导出坐标与事件',self.export_data)
        self.export_btn=self.button(outputs,'导出当前图层视频',self.export_movie)
        self.plot_btn=self.button(outputs,'导出二维落点图',lambda:self.export_distribution_ui(False))
        self.sync_btn=self.button(outputs,'导出轨迹＋落点同步视频',lambda:self.export_distribution_ui(True))
        self.export_btn.setToolTip('导出使用当前图层开关；缩放仅影响预览，导出保留完整画面。审核入库仍保存原视频。')
        self.approve_btn=self.button(sl,'审核通过 → video_success',self.approve_current,'primary')
        self.library_btn=self.button(outputs,'打开合格视频文件夹',self.open_library)
        self.training_btn=self.button(outputs,'生成已审核训练清单',self.export_training)
        sl.addStretch()
        self.side_scroll=QScrollArea();self.side_scroll.setWidgetResizable(True);self.side_scroll.setFixedWidth(240)
        self.side_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff);self.side_scroll.setWidget(side)
        left=QWidget();left.setFixedWidth(240);left_layout=QVBoxLayout(left);left_layout.setContentsMargins(0,0,0,0)
        left_layout.addWidget(self.side_scroll,1)
        actions=QHBoxLayout();sl.removeWidget(self.run_btn);sl.removeWidget(self.cancel_btn)
        self.cancel_btn.setText('取消任务');actions.addWidget(self.run_btn,1);actions.addWidget(self.cancel_btn,1)
        left_layout.addLayout(actions);sl.removeWidget(self.approve_btn);left_layout.addWidget(self.approve_btn);body.addWidget(left)
        center=QVBoxLayout();center.setSpacing(12);body.addLayout(center,1)
        video_panel=panel();vl=QVBoxLayout(video_panel);vl.setContentsMargins(12,12,12,12)
        top=QHBoxLayout();self.video_info=QLabel('视频预览');self.video_info.setObjectName('section');top.addWidget(self.video_info);top.addStretch()
        vl.addLayout(top)
        layer_row=QHBoxLayout();layer_row.addWidget(QLabel('显示'))
        self.layer_checks={}
        for key,title in [('trail','轨迹'),('ball','球心'),('bounces','落点'),('strokes','击球'),('terminal','出界 / 拦网'),('boundary','球台边界')]:
            check=QCheckBox(title);check.setChecked(True);check.toggled.connect(self.refresh_frame)
            self.layer_checks[key]=check;layer_row.addWidget(check)
        self.overlay=self.layer_checks['trail']
        layer_row.addStretch()
        self.clean_view=QCheckBox('纯视频');self.clean_view.toggled.connect(self.refresh_frame);layer_row.addWidget(self.clean_view)
        vl.addLayout(layer_row)
        self.view=VideoView();self.view.clicked.connect(self.video_click);self.view.corner_moved.connect(self.drag_corner);vl.addWidget(self.view,1)
        self.zoom_label=QLabel('100%');top.addWidget(self.zoom_label)
        self.pan_button=QPushButton('拖动画面');self.pan_button.setCheckable(True);self.pan_button.setFixedWidth(84)
        self.pan_button.setToolTip('开启后按住左键拖动画面；关闭后可标点。右键、中键始终可拖动。')
        self.pan_button.toggled.connect(self.view.set_pan_mode);top.addWidget(self.pan_button)
        for title,callback in [('－',lambda:self.view.zoom_at(1/1.25)),('＋',lambda:self.view.zoom_at(1.25)),('复位',self.view.reset_zoom)]:
            button=QPushButton(title);button.setFixedWidth(48);button.clicked.connect(callback);top.addWidget(button)
        self.view.zoom_changed.connect(lambda value:self.zoom_label.setText(f'{value*100:.0f}%'))
        transport=QHBoxLayout();self.back_btn=QPushButton('后退');self.back_btn.clicked.connect(lambda:self.step_frames(-1));transport.addWidget(self.back_btn)
        self.play_btn=QPushButton('播放');self.play_btn.clicked.connect(self.toggle_play);transport.addWidget(self.play_btn)
        self.play_btn.setToolTip('播放预览按时间同步，最高30帧/秒；暂停逐帧、分析与导出保留原始全部帧。')
        self.next_btn=QPushButton('前进');self.next_btn.clicked.connect(lambda:self.step_frames(1));transport.addWidget(self.next_btn)
        transport.addWidget(QLabel('每次'))
        self.step_input=QSpinBox();self.step_input.setRange(1,10000);self.step_input.setValue(1);self.step_input.setSuffix(' 帧');self.step_input.setToolTip('输入 2、3 或其他步长；前进、后退及键盘左右键都使用此值。');self.step_input.setMaximumWidth(105);transport.addWidget(self.step_input)
        self.playback_speed=QComboBox()
        for text,value in [('0.25 倍',.25),('0.5 倍',.5),('1 倍',1.),('2 倍',2.)]:self.playback_speed.addItem(text,value)
        self.playback_speed.setCurrentIndex(2);self.playback_speed.currentIndexChanged.connect(self.change_speed);transport.addWidget(self.playback_speed)
        vl.addLayout(transport)
        seek_row=QHBoxLayout();seek_row.addWidget(QLabel('跳到帧'))
        self.goto_input=QSpinBox();self.goto_input.setRange(0,0);self.goto_input.setMaximumWidth(110);seek_row.addWidget(self.goto_input)
        self.goto_btn=QPushButton('跳转');self.goto_btn.clicked.connect(lambda:self.seek(self.goto_input.value()));seek_row.addWidget(self.goto_btn)
        self.clock=QLabel('0 / 0 秒');self.clock.setObjectName('muted');self.clock.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Preferred)
        seek_row.addWidget(self.clock,1);vl.addLayout(seek_row)
        self.range_start_btn=QPushButton('设起点');self.range_start_btn.clicked.connect(lambda:self.set_range_marker(True));seek_row.addWidget(self.range_start_btn)
        self.range_end_btn=QPushButton('设终点');self.range_end_btn.clicked.connect(lambda:self.set_range_marker(False));seek_row.addWidget(self.range_end_btn)
        self.slider=QSlider(Qt.Horizontal);self.slider.setRange(0,0);self.slider.sliderPressed.connect(self.pause);self.slider.sliderReleased.connect(lambda:self.seek(self.slider.value()));vl.addWidget(self.slider)
        self.center_split=QSplitter(Qt.Vertical);self.center_split.setChildrenCollapsible(False)
        self.center_split.addWidget(video_panel);center.addWidget(self.center_split,1)
        event_panel=panel();event_panel.setStyleSheet('QPushButton { padding:6px 8px; }')
        el=QVBoxLayout(event_panel);el.setContentsMargins(12,12,12,12)
        event_title=QHBoxLayout();self.section(event_title,'事件复核');event_title.addStretch()
        self.prev_event_btn=QPushButton('上一事件');self.prev_event_btn.clicked.connect(lambda:self.jump_event(-1));event_title.addWidget(self.prev_event_btn)
        self.next_event_btn=QPushButton('下一事件');self.next_event_btn.clicked.connect(lambda:self.jump_event(1));event_title.addWidget(self.next_event_btn)
        self.undo_btn=QPushButton('撤销');self.undo_btn.setToolTip('撤销最近10次标注操作（Ctrl+Z）');self.undo_btn.clicked.connect(self.undo_annotation);event_title.addWidget(self.undo_btn)
        self.recalc_btn=QPushButton('重算候选');self.recalc_btn.clicked.connect(self.refresh_bounce_candidates);event_title.addWidget(self.recalc_btn)
        el.addLayout(event_title);review_tools=self.fold(el,'标注与筛选');eh=QHBoxLayout()
        self.edit_btn=QPushButton('修正球心');self.edit_btn.clicked.connect(self.start_edit);eh.addWidget(self.edit_btn)
        self.add_btn=QPushButton('新增事件');self.add_btn.clicked.connect(self.add_event);eh.addWidget(self.add_btn)
        self.confirm_btn=QPushButton('确认 / 撤销');self.confirm_btn.clicked.connect(self.confirm_event);eh.addWidget(self.confirm_btn)
        self.delete_btn=QPushButton('删除');self.delete_btn.clicked.connect(self.delete_event);eh.addWidget(self.delete_btn);review_tools.addLayout(eh)
        edit_row=QHBoxLayout()
        self.event_edit_btn=QPushButton('编辑选中事件');self.event_edit_btn.clicked.connect(self.edit_selected_event);edit_row.addWidget(self.event_edit_btn)
        self.event_point_btn=QPushButton('点击画面改落点');self.event_point_btn.clicked.connect(self.start_event_point_edit);edit_row.addWidget(self.event_point_btn)
        self.hide_ball_btn=QPushButton('不可见区间…');self.hide_ball_btn.clicked.connect(self.edit_invisible_range);edit_row.addWidget(self.hide_ball_btn)
        review_tools.addLayout(edit_row)
        filters=QHBoxLayout();self.event_filter=QComboBox()
        for title,key in [('全部事件','all'),('待复核','pending'),('已确认','confirmed'),('触台落点','bounce'),('击球接触','stroke'),('出界','out'),('拦网','net'),('近端落点','near'),('远端落点','far'),('已隐藏','hidden')]:self.event_filter.addItem(title,key)
        self.event_filter.currentIndexChanged.connect(self.filter_events);filters.addWidget(self.event_filter)
        self.event_search=QLineEdit();self.event_search.setPlaceholderText('编号 / 帧号');self.event_search.setClearButtonEnabled(True)
        self.event_search.setMaximumWidth(125);self.event_search.textChanged.connect(self.filter_events);filters.addWidget(self.event_search)
        self.nearby_events=QCheckBox('当前附近');self.nearby_events.setToolTip('仅显示当前帧前后2秒内的事件');self.nearby_events.toggled.connect(self.filter_events);filters.addWidget(self.nearby_events)
        clear_filter=QPushButton('清除');clear_filter.setFixedWidth(54);clear_filter.clicked.connect(self.clear_event_filters);filters.addWidget(clear_filter);filters.addStretch();review_tools.addLayout(filters)
        self.events=QTableWidget(0,7);self.events.setHorizontalHeaderLabels(['编号','帧号','秒','事件','坐标 / 米','审核','显示'])
        self.events.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch);self.events.verticalHeader().hide()
        self.events.verticalHeader().setDefaultSectionSize(34)
        self.events.setSelectionBehavior(QTableWidget.SelectRows);self.events.setEditTriggers(QTableWidget.NoEditTriggers)
        self.events.setSelectionMode(QTableWidget.ExtendedSelection)
        self.events.currentCellChanged.connect(lambda row,col,*args:self.event_clicked(row,col))
        self.events.itemSelectionChanged.connect(self.update_selection_details)
        self.events.cellDoubleClicked.connect(lambda row,col:self.toggle_selected_visibility() if col==6 else self.edit_selected_event())
        self.events.setContextMenuPolicy(Qt.CustomContextMenu);self.events.customContextMenuRequested.connect(self.event_context_menu)
        self.events.setToolTip('Ctrl/Shift 多选；双击“显示”切换隐藏；右键可批量确认、隐藏或复制坐标。')
        self.events.setAlternatingRowColors(True);el.addWidget(self.events)
        self.center_split.addWidget(event_panel);self.center_split.setSizes([520,190])
        right=panel();right.setFixedWidth(265);rl=QVBoxLayout(right);rl.setContentsMargins(14,16,14,16)
        rl.setSizeConstraint(QLayout.SetMinimumSize)
        self.section(rl,'球台落点分布');hint=QLabel('先标定球台，候选落点才会映射到这里。');hint.setObjectName('muted');hint.setWordWrap(True);rl.addWidget(hint)
        self.table=TableView();rl.addWidget(self.table,1)
        self.table.event_selected.connect(self.select_number);self.table.hide_requested.connect(self.hide_number);self.table.restore_requested.connect(self.restore_all_points)
        map_controls=QHBoxLayout();self.show_numbers=QCheckBox('编号');self.show_numbers.setChecked(True)
        self.show_numbers.toggled.connect(self.toggle_numbers);map_controls.addWidget(self.show_numbers)
        self.map_count=QLabel('');self.map_count.setWordWrap(True);self.map_count.setMinimumWidth(0);self.map_count.setObjectName('muted');map_controls.addWidget(self.map_count,1);rl.addLayout(map_controls)
        visibility_controls=QHBoxLayout();self.hide_selected_btn=QPushButton('隐藏所选');self.hide_selected_btn.clicked.connect(self.toggle_selected_visibility);visibility_controls.addWidget(self.hide_selected_btn)
        self.restore_points_btn=QPushButton('恢复全部');self.restore_points_btn.clicked.connect(self.restore_all_points);visibility_controls.addWidget(self.restore_points_btn);rl.addLayout(visibility_controls)
        self.selection_details=QLabel('点击落点查看编号与事件；重叠点可重复点击切换。');self.selection_details.setWordWrap(True);self.selection_details.setObjectName('muted');rl.addWidget(self.selection_details)
        self.heat=QCheckBox('热力分布（含未确认候选）');self.heat.toggled.connect(self.change_heat);rl.addWidget(self.heat)
        self.stats=QLabel('检测帧  —\n候选触台  —\n人工确认  —\n处理速度  —');self.stats.setStyleSheet('line-height:1.8; color:#b5c6da;');rl.addWidget(self.stats)
        diagnostics=self.fold(rl,'回合复核提示')
        self.hints_list=QListWidget();self.hints_list.setMaximumHeight(120);self.hints_list.itemClicked.connect(self.hint_clicked);diagnostics.addWidget(self.hints_list)
        right.setFixedWidth(245)
        self.right_scroll=QScrollArea();self.right_scroll.setFixedWidth(265);self.right_scroll.setWidgetResizable(True)
        self.right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff);self.right_scroll.setWidget(right);body.addWidget(self.right_scroll)
        footer=QHBoxLayout();self.status=QLabel('就绪 · 支持 MP4、MOV、AVI 等本地视频');self.status.setObjectName('muted');footer.addWidget(self.status,1)
        self.progress=QProgressBar();self.progress.setRange(0,100);self.progress.setValue(0);self.progress.setFixedWidth(230);footer.addWidget(self.progress);offline_layout.addLayout(footer)
        self.status.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Preferred);self.status.setMinimumWidth(0)
        self.live_panel=LivePanel(VideoView,lambda:(self.model_path,self.threshold.value(),self.acceleration.currentData()))
        self.live_panel.stopped.connect(lambda:self.close() if self.pending_close else None)
        self.pages.addWidget(self.live_panel)
        style=STYLE.replace('__DOWN__',asset_path('assets/chevron-down.svg').as_posix()).replace('__UP__',asset_path('assets/chevron-up.svg').as_posix())
        self.setStyleSheet(style);self.update_controls()
        self.model_preset.currentIndexChanged.connect(self.choose_preset)
        self.review_shortcuts=[]
        for key,callback in [('Left',lambda:self.step_frames(-1)),('Right',lambda:self.step_frames(1)),('Space',self.toggle_play)]:
            shortcut=QShortcut(QKeySequence(key),self);shortcut.activated.connect(callback);self.review_shortcuts.append(shortcut)
        self.escape_shortcut=QShortcut(QKeySequence('Escape'),self);self.escape_shortcut.activated.connect(self.cancel_annotation)
        self.review_shortcuts.append(self.escape_shortcut)
        for key,callback in [('Ctrl+Z',self.undo_annotation),('Alt+Left',lambda:self.jump_event(-1)),('Alt+Right',lambda:self.jump_event(1)),
            ('Alt+H',self.toggle_selected_visibility),('Alt+Shift+H',self.restore_all_points),('Ctrl+Return',lambda:self.set_selected_confirmation(True))]:
            shortcut=QShortcut(QKeySequence(key),self);shortcut.activated.connect(callback);self.review_shortcuts.append(shortcut)

    def fold(self, parent, title):
        trigger=QPushButton('＋  '+title)
        trigger.setObjectName('fold');trigger.setCheckable(True);parent.addWidget(trigger)
        content=QWidget();inner=QVBoxLayout(content);inner.setContentsMargins(0,4,0,3);inner.setSpacing(6)
        parent.addWidget(content);content.hide()
        trigger.toggled.connect(content.setVisible)
        trigger.toggled.connect(lambda opened:trigger.setText(('－  ' if opened else '＋  ')+title))
        return inner

    def switch_mode(self, page):
        current=self.pages.currentIndex()
        if page!=current and (self.task or self.live_panel.worker):
            (self.status if self.task else self.live_panel.status).setText('请先停止当前任务，再切换工作区。')
            page=current
        self.offline_tab.setChecked(page==0);self.live_tab.setChecked(page==1)
        if page==current:return
        self.pause()
        self.pages.setCurrentIndex(page)
        self.hero_title.setText('实时观测' if page else '比赛复盘')
        self.hero_caption.setText('固定机位 · 随画面追踪球路与候选事件' if page else '导入固定机位视频，分析轨迹、复核落点并导出。')
        for shortcut in self.review_shortcuts:shortcut.setEnabled(page==0)

    def section(self,layout,text):
        label=QLabel(text);label.setObjectName('section');layout.addWidget(label)

    def button(self,layout,text,callback,name=None):
        button=QPushButton(text);button.clicked.connect(callback)
        if name:button.setObjectName(name)
        layout.addWidget(button);return button

    def error(self,text):
        QMessageBox.warning(self,'操作未完成',text)

    def safe_to_replace(self):
        if not self.dirty:return True
        answer=QMessageBox.question(self,'保存修改','当前分析有未保存的修改，是否先保存？',QMessageBox.Save|QMessageBox.Discard|QMessageBox.Cancel,QMessageBox.Save)
        if answer==QMessageBox.Cancel:return False
        if answer==QMessageBox.Save:return self.save()
        return True

    def open_video(self):
        if not self.safe_to_replace():return
        path,_=QFileDialog.getOpenFileName(self,'选择比赛视频',str(workspace_root()/'video'),'视频 (*.mp4 *.mov *.avi *.mkv *.webm *.ogv *.m4v *.mts *.m2ts *.ts *.wmv *.flv *.mxf *.mpeg *.mpg);;所有文件 (*)')
        if path and self.load_video(path) and self.auto_on_load.isChecked():self.auto_calibrate()

    def load_video(self,path):
        path=resolve_video(path)
        try:cap=VideoReader(path)
        except Exception as error:self.error(str(error));return False
        fps=cap.get(cv2.CAP_PROP_FPS);total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok,frame=cap.read()
        if not ok or not math.isfinite(fps) or fps<=0 or total<=0:
            cap.release();self.error('视频帧率、帧数无效或无法解码。');return False
        self.pause()
        if self.cap:self.cap.release()
        self.cap=cap;self.video=str(Path(path).resolve());self.fps=fps;self.total=total
        self.project=None;self.corners=[];self.dirty=False;self.calibrating=False;self.editing=False;self.calibration_info={}
        self.edit_event_ref=None;self.pending_add_event=False
        self.history=AnnotationHistory();self.range_start=None;self.range_end=None
        self.selected_number=None;self.clear_event_filters()
        self.range_start_btn.setText('设起点');self.range_end_btn.setText('设终点')
        details=media_details(path,sha256=digest(path) if Path(path).name=='demo.mp4' else None)
        camera={'oblique':'斜视角','side':'侧视角'}.get(details.get('camera'),'')
        self.slider.setRange(0,total-1);self.goto_input.setRange(0,total-1)
        self.filename.setText(f"{details['id']} · {total/fps:.1f} 秒"+(f' · {camera}' if camera else ''))
        self.filename.setToolTip(details.get('original_name',Path(path).name))
        self.video_info.setText(f'{frame.shape[1]} × {frame.shape[0]}  ·  {fps:.2f} FPS')
        self.frame_index=0;self.raw_frame=frame;self.refresh_frame();self.refresh_events();self.update_controls()
        self.view.reset_zoom()
        self.pan_button.setChecked(False)
        self.status.setText('视频已载入 · 建议先标定球台，再开始分析');return True

    def open_project(self):
        if not self.safe_to_replace():return
        path,_=QFileDialog.getOpenFileName(self,'打开项目',str(app_root()),'分析项目 (*.json)')
        if not path:return
        try:
            project=json.loads(Path(path).read_text(encoding='utf-8'));validate_project(project)
            local=Path(path).parent/project['video']
            project['video']=str(local if not Path(project['video']).is_absolute() and local.exists() else resolve_video(project['video']))
            if not Path(project['video']).exists():
                replacement,_=QFileDialog.getOpenFileName(self,'原视频已移动，请重新选择')
                if not replacement:return
                project['video']=replacement
            if not self.load_video(project['video']):return
            decoded=project.get('decode_info',{})
            compatible_count=project['frame_count']==self.total or (decoded.get('reported_frames')==self.total and decoded.get('decoded_frames')==project['frame_count'])
            if not compatible_count or project['width']!=self.raw_frame.shape[1] or project['height']!=self.raw_frame.shape[0] or abs(project['fps']-self.fps)>.05:
                raise ValueError('项目与所选视频的帧数、尺寸或帧率不匹配。')
            self.total=project['frame_count'];self.cap.set_timeline([p['time'] for p in project['tracks']]);self.slider.setRange(0,self.total-1);self.goto_input.setRange(0,self.total-1)
            self.project=project;self.corners=project.get('corners',[]);self.threshold.setValue(project['threshold'])
            self.motion_filter.setChecked(project.get('motion_filter_enabled',True))
            migrated=project.get('event_algorithm')!=ALGORITHM
            if migrated:self.recalculate_events()
            self.refresh_events();self.refresh_frame();self.update_controls()
            self.status.setText('旧项目的未确认候选已按新版重算，已确认事件保留，请检查并保存' if migrated else '分析项目已载入')
            if not project.get('motion_filter_enabled',False):self.status.setText('已载入原有分析；持续静止杂点过滤需启用后重新分析，现有人工标注未改动')
        except Exception as e:self.error(str(e))

    def choose_model(self):
        path,_=QFileDialog.getOpenFileName(self,'选择 ONNX 检测模型',str(app_root()),'ONNX 模型 (*.onnx)')
        if path:
            self.model_path=Path(path);self.model_label.setText(self.model_path.name);self.threshold.setValue(model_threshold(self.model_path))
            self.model_preset.blockSignals(True)
            self.model_preset.setCurrentIndex(self.model_preset.findData(str(self.model_path)))
            self.model_preset.blockSignals(False)

    def choose_preset(self,index):
        path=self.model_preset.itemData(index)
        if not path:return
        self.model_path=Path(path);self.model_label.setText(self.model_path.name)
        self.model_label.setToolTip(str(self.model_path));self.threshold.setValue(model_threshold(self.model_path))
        self.status.setText('已切换模型方案；已有结果不会改变，重新分析后使用新模型和推荐阈值')

    def seek(self,index,force=False):
        if self.cap is None or (self.task and not force):return
        if self.timer.isActive():self.pause()
        index=max(0,min(self.total-1,int(index)))
        if int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))!=index:
            if not self.cap.set(cv2.CAP_PROP_POS_FRAMES,index):self.pause();self.status.setText('该帧超出可读取范围；完成分析后将按实际帧数校正进度条');return
        ok,frame=self.cap.read()
        if not ok:self.pause();self.status.setText(f'无法解码第 {index} 帧');return
        self.frame_index=index;self.raw_frame=frame;self.refresh_frame()
        if self.project and self.nearby_events.isChecked():self.refresh_events()

    def step_frames(self,direction):
        if not self.video or self.task:return
        self.pause();self.seek(self.frame_index+direction*self.step_input.value())

    def display_options(self):
        return {key:check.isChecked() and not self.clean_view.isChecked() for key,check in self.layer_checks.items()}

    def render_options(self):
        return {**self.display_options(),'hidden_event_numbers':list(hidden_numbers(self.project or {})),
            'visible_event_numbers':[self.project['events'][i]['event_number'] for i in self.visible_rows] if self.project else None,
            'event_numbers':self.show_numbers.isChecked(),'selected_event_number':self.selected_number}

    def refresh_frame(self,*args):
        if self.raw_frame is None:return
        options=self.display_options();show_boundary=options['boundary'] or self.calibrating
        render_options=self.render_options()
        image=render_frame(self.raw_frame,self.project,self.frame_index,options={**render_options,'boundary':False}) if self.project else self.raw_frame.copy()
        for i,(x,y) in enumerate(self.corners if show_boundary else []):
            cv2.circle(image,(round(x),round(y)),9,(100,255,100),-1)
            cv2.putText(image,str(i+1),(round(x)+12,round(y)-12),cv2.FONT_HERSHEY_SIMPLEX,1,(100,255,100),2)
        if show_boundary and len(self.corners)==4:cv2.polylines(image,[np.array(self.corners,np.int32)],True,(100,255,100),2,cv2.LINE_AA)
        allowed=set(render_options['visible_event_numbers'] or []) if self.project else None
        if self.table.show_points!=options['bounces'] or self.table.allowed!=allowed:
            self.table.show_points=options['bounces'];self.table.allowed=allowed
            self.table.prepare_layout();self.table.update();self.update_map_count()
        self.view.corners=self.corners;self.view.allow_drag=show_boundary and not(self.task or self.calibrating or self.editing or self.edit_event_ref or self.pending_add_event)
        self.view.annotation_mode=bool(self.calibrating or self.editing or self.edit_event_ref or self.pending_add_event)
        self.view.set_frame(image);self.slider.setValue(self.frame_index)
        self.goto_input.setValue(self.frame_index)
        timestamp=self.project['tracks'][self.frame_index]['time'] if self.project and self.frame_index<len(self.project['tracks']) else self.cap.last_timestamp
        duration=self.project.get('duration_seconds',self.project['tracks'][-1]['time']) if self.project else (self.total-1)/self.fps
        self.clock.setText(f'{timestamp:.3f} / {duration:.3f} 秒 · 帧 {self.frame_index}')
        self.clock.setToolTip(self.clock.text())

    def toggle_play(self):
        if self.task:return
        if self.timer.isActive():self.pause()
        elif self.cap:
            if self.frame_index>=self.total-1:self.seek(0)
            times=[p['time'] for p in self.project['tracks']] if self.project else None
            self.playback=PlaybackDecoder(self.video,self.frame_index,self.fps,times,self.playback_speed.currentData(),size=self.raw_frame.shape[1::-1])
            self.play_wall=None;self.timer.start(10);self.play_btn.setText('暂停')

    def playback_interval(self):
        dt=1/self.fps
        if self.project and self.frame_index+1<len(self.project['tracks']):
            dt=self.project['tracks'][self.frame_index+1]['time']-self.project['tracks'][self.frame_index]['time']
        return max(1,round(dt*1000/self.playback_speed.currentData()))

    def change_speed(self,*args):
        if self.timer.isActive():self.pause();self.toggle_play()

    def pause(self):
        self.timer.stop();self.play_btn.setText('播放')
        if self.playback:self.playback.stop();self.playback=None
        self.play_wall=None

    def advance(self):
        worker=self.playback
        if worker is None:return
        if not worker.ready.is_set():return
        if self.play_wall is None:self.play_wall=time.perf_counter()
        target=worker.start_time+(time.perf_counter()-self.play_wall)*self.playback_speed.currentData()
        item=worker.pop_due(target)
        if item:
            index,timestamp,frame=item;self.frame_index=min(index,self.total-1);self.raw_frame=frame
            if self.project and self.nearby_events.isChecked() and time.perf_counter()-self.last_nearby_refresh>.25:
                self.last_nearby_refresh=time.perf_counter();self.refresh_events()
            self.refresh_frame()
        if worker.done and worker.empty():
            error=worker.error;self.pause()
            if error:self.status.setText('播放解码中断，可暂停后逐帧定位；'+error)

    def auto_calibrate(self):
        if self.raw_frame is None or self.task:return
        self.pause();self.calibrating=False;self.editing=False
        self.launch_task('calibrate',dict(frame=self.raw_frame.copy(),video=self.video,frame_index=self.frame_index,
                                        fps=self.fps,acceleration=self.acceleration.currentData()))

    def recalculate_events(self):
        if not self.project:return
        ensure_numbers(self.project);previous=self.project['events']
        preserved=[e for e in self.project['events'] if e['confirmed'] or e['source']=='manual']
        candidates,hints=analyze_events([TrackPoint(**p) for p in self.project['tracks']],self.fps,self.corners,
            image_size=(self.project['width'],self.project['height']),break_frames=self.project.get('break_frames',[]))
        # Preserve reviewed and manual events, replacing only unreviewed proposals.
        excluded=preserved+self.project.get('rejected_events',[])
        candidates=[e for e in candidates if not any(event_kind(e)==event_kind(p) and abs(e['frame']-p['frame'])<=2 for p in excluded)]
        reuse_numbers([e for e in previous if e not in preserved],candidates)
        for e in preserved:e['table_xy']=map_point(e['x'],e['y'],self.corners) if event_kind(e)=='bounce' else None
        self.project['events']=sorted(preserved+candidates,key=lambda e:e['frame'])
        self.project['event_algorithm']=ALGORITHM;self.project['review_hints']=hints
        invalidate_review(self.project)
        self.dirty=True

    def refresh_bounce_candidates(self):
        if self.project and not self.task:
            self.remember_annotation()
            self.recalculate_events();self.refresh_events();self.refresh_frame()
            self.status.setText('已重算触台、出界和拦网候选，保留人工修正与误检删除记录')

    def apply_corners(self,corners,info=None):
        homography(corners)
        if self.project:self.remember_annotation()
        self.corners=[list(p) for p in corners]
        self.calibration_info=info or {'method':'manual'}
        if self.project:
            log_edit(self.project,'table_calibration',self.project.get('corners'),self.corners)
            self.project['corners']=self.corners;self.project['calibration_info']=self.calibration_info
            self.recalculate_events()
        self.refresh_events();self.refresh_frame()

    def rotate_calibration(self):
        if len(self.corners)==4 and not self.task:
            self.apply_corners(self.corners[1:]+self.corners[:1],{'method':'orientation_adjusted'})
            self.status.setText('球台方向已旋转 90°，请检查远近端及落点图')

    def drag_corner(self,index,x,y,finished):
        if self.task or len(self.corners)!=4:return
        if self.corner_backup is None:self.corner_backup=[p[:] for p in self.corners]
        self.corners=[p[:] for p in self.corners];self.corners[index]=[x,y]
        if finished:
            try:self.apply_corners(self.corners,{'method':'manually_refined'})
            except ValueError:
                self.corners=self.corner_backup;self.status.setText('角点不能交叉或共线，本次拖动已撤销')
            else:self.status.setText('球台边界已微调，未确认触台候选已重新筛选')
            self.corner_backup=None
        self.refresh_frame()

    def start_calibration(self):
        if not self.video:return
        self.pan_button.setChecked(False)
        self.clean_view.setChecked(False);self.layer_checks['boundary'].setChecked(True)
        self.pause();self.corners=[];self.calibrating=True;self.editing=False
        self.status.setText('依次点击：远端左角 → 远端右角 → 近端右角 → 近端左角');self.refresh_frame()

    def clear_calibration(self):
        if self.project:self.remember_annotation()
        if self.project:log_edit(self.project,'table_cleared',self.project.get('corners'),[])
        self.corners=[];self.calibrating=False
        if self.project:
            self.project['corners']=[]
            for e in self.project['events']:e['table_xy']=None
            self.dirty=True
        self.refresh_events();self.refresh_frame();self.status.setText('球台标定已清除')

    def video_click(self,x,y):
        if self.task:return
        if self.calibrating:
            self.corners.append([x,y])
            if len(self.corners)==4:
                try:homography(self.corners)
                except ValueError as e:
                    self.corners=[];self.error(str(e));return
                self.calibrating=False
                self.apply_corners(self.corners)
                self.status.setText('球台标定完成');self.refresh_events()
            self.refresh_frame()
        elif self.edit_event_ref is not None and self.project:
            event=self.edit_event_ref;self.edit_event_ref=None
            self.apply_event_changes(event,dict(kind=event_kind(event),frame=self.frame_index,x=x,y=y,confirmed=True))
            self.status.setText('事件落点已人工修正，审核入库后会保留此标签')
        elif self.pending_add_event and self.project:
            self.pending_add_event=False
            self.show_event_dialog(None,dict(frame=self.frame_index,time=self.frame_index/self.fps,x=x,y=y,kind='bounce',type='bounce_candidate',confirmed=False,source='manual'))
        elif self.editing and self.project:
            self.remember_annotation()
            p=self.project['tracks'][self.frame_index];before=copy.deepcopy(p);p.update(x=x,y=y,confidence=1.,source='manual',candidate_x=None,candidate_y=None)
            for e in self.project['events']:
                if e['frame']==self.frame_index:
                    e.update(x=x,y=y,table_xy=map_point(x,y,self.corners) if event_kind(e)=='bounce' else None,
                        origin_source=e.get('origin_source',e['source']),source='manual')
            log_edit(self.project,'ball_position',before,p)
            self.editing=False;self.dirty=True;self.refresh_frame();self.refresh_events();self.status.setText('本帧球心已修正；已有事件同步更新')

    def start_edit(self):
        if self.project:
            self.pan_button.setChecked(False)
            self.pause();self.editing=True;self.calibrating=False;self.edit_event_ref=None;self.pending_add_event=False
            self.status.setText('点击画面中的真实小球中心，修正本帧坐标');self.refresh_frame()

    def mark_ball_invisible(self):
        self.apply_invisible_range(self.frame_index,self.frame_index)

    def set_range_marker(self,start):
        if not self.video or self.task:return
        self.pause()
        if start:self.range_start=self.frame_index;self.range_start_btn.setText(f'起 {self.frame_index}')
        else:self.range_end=self.frame_index;self.range_end_btn.setText(f'止 {self.frame_index}')
        self.status.setText('区间起止已设置，点击“不可见区间…”检查并应用（包含首尾帧）')

    def edit_invisible_range(self):
        if not self.project or self.task:return
        self.pause()
        start=self.frame_index if self.range_start is None else self.range_start
        end=self.frame_index if self.range_end is None else self.range_end
        dialog=InvisibleRangeDialog(self.total,self.frame_index,start,end,self)
        if dialog.exec()==QDialog.Accepted:
            self.apply_invisible_range(dialog.start.value(),dialog.end.value(),dialog.remove_events.isChecked())

    def apply_invisible_range(self,start,end,remove_events=False):
        if not self.project or self.task:return
        if not 0<=start<=end<self.total:self.error('区间无效，请检查起始帧和结束帧。');return
        self.pause();self.cancel_annotation();self.remember_annotation()
        count=mark_invisible_range(self.project,start,end,remove_events)
        self.dirty=True;self.refresh_events();self.refresh_frame()
        self.status.setText(f'已标记第 {start}–{end} 帧球不可见，共 {count} 帧；Ctrl+Z 可撤销')

    def remember_annotation(self):
        if self.project:self.history.remember(self.project);self.undo_btn.setEnabled(True)

    def undo_annotation(self):
        if not self.project or self.task:return
        restored=self.history.undo(self.project)
        if restored is None:return
        self.pause();self.project=restored;self.corners=copy.deepcopy(restored.get('corners',[]))
        self.calibration_info=copy.deepcopy(restored.get('calibration_info',{}));self.dirty=True
        self.cancel_annotation();self.refresh_events();self.update_controls()
        self.status.setText('已撤销上一次标注操作；已入库文件不受影响，修改后需重新审核')

    def jump_event(self,direction):
        if not self.project or self.task:return
        ordered=sorted([self.project['events'][i] for i in self.filtered_indices()],key=lambda e:e['frame'],reverse=direction<0)
        target=next((e for e in ordered if (e['frame']-self.frame_index)*direction>0),None)
        if target is not None:
            self.select_number(target['event_number'])
        else:self.status.setText('已到该方向的最后一个事件')

    def start_event_point_edit(self):
        self.pan_button.setChecked(False)
        row=self.selected_event()
        if row is None:self.error('请先在事件列表中选择一个落点或事件。');return
        self.pause();self.edit_event_ref=self.project['events'][row];self.editing=False;self.calibrating=False
        self.status.setText('可先跳到正确帧，再点击画面中的实际落点；也可用“编辑选中事件”改类型和帧号');self.refresh_frame()

    def edit_selected_event(self):
        row=self.selected_event()
        if row is not None:self.show_event_dialog(self.project['events'][row])

    def show_event_dialog(self,event,initial=None):
        self.pause();dialog=EventDialog(event or initial,self.project,self.frame_index,self)
        if dialog.exec()==QDialog.Accepted:
            try:self.apply_event_changes(event,dialog.values())
            except ValueError as exc:self.error(str(exc))

    def apply_event_changes(self,event,values):
        if not self.project:return
        frame=int(values['frame']);kind=values['kind'];x=float(values['x']);y=float(values['y'])
        if kind not in ('bounce','out','net','stroke') or not 0<=frame<self.total or not(0<=x<self.project['width'] and 0<=y<self.project['height']):
            raise ValueError('事件帧号、类型或坐标无效。')
        if any(e is not event and e['frame']==frame and event_kind(e)==kind for e in self.project['events']):
            raise ValueError('这一帧已存在同类型事件，请编辑现有事件。')
        self.remember_annotation()
        before=copy.deepcopy(event);target=event if event is not None else {}
        if before is not None and (before['frame']!=frame or event_kind(before)!=kind):
            self.project.setdefault('rejected_events',[]).append(dict(before,superseded=True))
        origin=target.get('origin_source',target.get('source','manual'))
        for key in ('fit_frame','fit_rmse_px','vertex_observed'):target.pop(key,None)
        target.update(frame=frame,time=self.project['tracks'][frame]['time'],x=x,y=y,kind=kind,source='manual',origin_source=origin,
            table_xy=map_point(x,y,self.corners) if kind=='bounce' else None,
            confirmed=bool(values.get('confirmed',True)),human_reviewed=bool(values.get('confirmed',True)))
        target['type']=kind if target['confirmed'] else kind+'_candidate'
        if event is None:self.project['events'].append(target)
        self.project['events'].sort(key=lambda e:e['frame'])
        log_edit(self.project,'event_added' if event is None else 'event_corrected',before,target)
        self.dirty=True;self.refresh_events();self.refresh_frame();self.select_number(target['event_number'])

    def approve_current(self):
        if not self.project or self.task:return
        if self.calibrating or self.editing or self.edit_event_ref or self.pending_add_event or len(self.corners)!=4:
            self.error('请先完成标注和球台标定；按 Esc 可以取消尚未完成的标注。');return
        self.pause();dialog=ApprovalDialog(self.project,self)
        if dialog.exec()==QDialog.Accepted:
            self.launch_task('approve',dict(project=copy.deepcopy(self.project),events_reviewed=True,ball_positions_reviewed=dialog.ball.isChecked()))

    def open_library(self):
        folder=approval_root();folder.mkdir(parents=True,exist_ok=True);QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def export_training(self):
        if not self.task:self.launch_task('training',{})

    def hint_clicked(self,item):
        value=item.data(Qt.UserRole)
        if value and self.project and not self.task:self.pause();self.seek((value['start']+value['end'])//2)

    def cancel_annotation(self):
        self.calibrating=False;self.editing=False;self.edit_event_ref=None;self.pending_add_event=False
        self.corners=copy.deepcopy(self.project.get('corners',[])) if self.project else []
        self.refresh_frame();self.status.setText('已取消尚未完成的标注，已保存的修正保留')

    def start_analysis(self):
        if not self.video or self.task:return
        if self.project and not self.safe_to_replace():return
        self.pause();self.calibrating=False;self.editing=False;self.edit_event_ref=None;self.pending_add_event=False
        self.launch_task('analyze',dict(video=self.video,model_path=self.model_path,
                                       threshold=self.threshold.value(),corners=self.corners,
                                       acceleration=self.acceleration.currentData(),motion_filter=self.motion_filter.isChecked()))

    def launch_task(self,kind,args):
        self.task=Task(kind,args);self.task.progress.connect(self.task_progress)
        self.task.result.connect(self.task_result);self.task.failed.connect(self.task_error)
        self.task.finished.connect(self.task_finished);self.progress.setValue(0)
        self.status.setText({'analyze':'正在分析视频…','calibrate':'正在自动识别球台…','export':'正在导出轨迹视频…',
            'approve':'正在保存人工审核视频与标签…','training':'正在校验审核数据并生成训练清单…','plot':'正在导出二维落点图…','sync':'正在导出轨迹与同步落点视频…'}[kind])
        if kind=='calibrate':self.progress.setRange(0,0)
        self.update_controls();self.task.start()

    def task_progress(self,done,total,frame,point):
        if self.task and self.task.cancel.is_set():
            self.status.setText('正在取消，等待当前计算结束…');return
        self.progress.setValue(round(done/max(1,total)*100))
        self.status.setText(f'正在处理 {done} / {total} 帧')
        if self.task and self.task.kind=='analyze':
            image=frame.copy()
            if point and point.x is not None:
                sy=image.shape[0]/self.raw_frame.shape[0];sx=image.shape[1]/self.raw_frame.shape[1]
                cv2.circle(image,(round(point.x*sx),round(point.y*sy)),max(3,round(9*sx)),(0,255,255),2)
            self.view.set_frame(image)

    def task_result(self,value):
        if isinstance(value,dict) and value.get('task_type')=='artifact':
            self.progress.setValue(100);self.status.setText('已导出：'+value['path']);self.status.setToolTip(self.status.text())
        elif isinstance(value,dict) and value.get('task_type')=='approve':
            for e in self.project['events']:e.update(confirmed=True,type=event_kind(e),human_reviewed=True)
            self.project['review']={'status':'approved','directory':value['directory']};self.dirty=False
            self.refresh_events();self.status.setText('已人工审核入库：'+value['directory']);self.status.setToolTip(self.status.text())
        elif isinstance(value,dict) and value.get('task_type')=='training':
            self.status.setText(f"训练清单已生成：{value['clips']} 段，{value['reviewed_ball_frames']} 个已审核球心帧")
            if not value['independent_validation']:self.status.setText(self.status.text()+'；尚缺独立来源作验证')
            self.status.setToolTip(value['file'])
        elif isinstance(value,dict) and value.get('task_type')=='calibration_unavailable':
            self.progress.setValue(0)
            self.status.setText('自动标台未成功，视频及原标定保留；请换一帧或手动选择四角')
            self.status.setToolTip(value['message'])
        elif isinstance(value,dict) and value.get('task_type')=='calibration':
            self.apply_corners(value['corners'],{k:v for k,v in value.items() if k not in ('corners','task_type')})
            count=len(value.get('verified_frames',[]))
            self.status.setText(f'已自动识别球台 · {count} 帧一致性检查，请核对绿线并微调四角')
            if value.get('review_required'):self.status.setText('仅当前帧识别成功，缺少多帧支持；请核对绿线或手动微调')
        elif isinstance(value,dict):
            self.history=AnnotationHistory()
            self.project=value;self.project['calibration_info']=self.calibration_info
            self.total=value['frame_count'];self.cap.set_timeline([p['time'] for p in value['tracks']])
            self.slider.setRange(0,self.total-1);self.goto_input.setRange(0,self.total-1)
            self.dirty=True;self.refresh_events();self.seek(0,force=True);self.progress.setValue(100)
            self.status.setText('分析完成 · 点击触台候选回看；橙色事件尚未确认')
            info=value.get('decode_info',{})
            if info.get('status')=='metadata_adjusted':self.status.setText(f'分析完成 · 已校正容器帧数估计，实际分析 {self.total} 帧')
            elif info.get('status') in ('partial','recovered'):self.status.setText(f'已分析可读取的 {self.total} 帧 · 原视频存在损坏或缺失，请复核可读区间')
        elif value:self.progress.setValue(100);self.status.setText('轨迹视频已导出（不含音频）')
        else:self.status.setText('任务已取消，原视频及已有项目保留');self.refresh_frame()

    def task_error(self,error):
        import os
        logdir=Path(os.environ.get('LOCALAPPDATA',str(app_root())))/'PingPongStudio/logs';logdir.mkdir(parents=True,exist_ok=True)
        (logdir/'last_error.txt').write_text(error,encoding='utf-8')
        self.status.setText('任务失败，详细信息已保存到运行日志');self.error(error[-1800:])

    def task_finished(self):
        if self.task:self.task.deleteLater()
        self.task=None;self.progress.setRange(0,100);self.update_controls();self.refresh_frame()
        if self.pending_close:self.close()

    def cancel_task(self):
        if self.task:
            self.task.cancel.set();self.task.requestInterruption();self.status.setText('正在取消，等待当前计算结束…')
            self.cancel_btn.setText('取消中…');self.cancel_btn.setEnabled(False)

    def filtered_indices(self):
        if not self.project:return []
        ensure_numbers(self.project);hidden=hidden_numbers(self.project);mode=self.event_filter.currentData()
        return [i for i,e in enumerate(self.project['events']) if (mode!='hidden' or e['event_number'] in hidden)
            and matches_filter(e,mode,self.event_search.text(),self.frame_index if self.nearby_events.isChecked() else None,self.fps)]

    def filter_events(self,*args):
        self.refresh_events();self.refresh_frame()

    def clear_event_filters(self):
        for widget in (self.event_filter,self.event_search,self.nearby_events):widget.blockSignals(True)
        self.event_filter.setCurrentIndex(0);self.event_search.clear();self.nearby_events.setChecked(False)
        for widget in (self.event_filter,self.event_search,self.nearby_events):widget.blockSignals(False)
        self.filter_events()

    def select_number(self,number):
        if not self.project or self.task:return
        event=next((e for e in self.project['events'] if e.get('event_number')==number),None)
        if event is None:return
        self.selected_number=number
        index=self.project['events'].index(event)
        if index not in self.visible_rows:self.clear_event_filters()
        if index in self.visible_rows:
            row=self.visible_rows.index(index)
            self.events.setCurrentCell(row,0);self.events.selectRow(row);self.events.scrollToItem(self.events.item(row,0))
        self.pause();self.seek(event['frame']);self.update_selection_details();self.refresh_frame()

    def selected_indices(self):
        return sorted({self.visible_rows[item.row()] for item in self.events.selectionModel().selectedRows() if item.row()<len(self.visible_rows)})

    def update_selection_details(self):
        row=self.selected_event();indices=self.selected_indices()
        if row is None:
            self.selected_number=None;self.table.selected=None
            self.selection_details.setText('点击落点或事件查看详情；Ctrl/Shift可多选。')
        else:
            e=self.project['events'][row];self.selected_number=e['event_number'];self.table.selected=self.selected_number
            xy=e.get('table_xy');where=f"{xy[0]:.3f}, {xy[1]:.3f} 米" if xy else '无台面坐标'
            name={'bounce':'触台','net':'拦网','out':'出界','stroke':'击球'}.get(event_kind(e),'事件')
            hidden=e['event_number'] in hidden_numbers(self.project)
            self.selection_details.setText(f"#{e['event_number']:03d} · {name} · 第{e['frame']}帧\n{e['time']:.3f} 秒 · {where}\n"+
                ('已确认' if e['confirmed'] else '待复核')+(' · 已隐藏' if hidden else ' · 显示中')+(f' · 已选{len(indices)}项' if len(indices)>1 else ''))
        hidden=hidden_numbers(self.project or {})
        all_hidden=bool(indices) and all(self.project['events'][i]['event_number'] in hidden for i in indices)
        self.hide_selected_btn.setText('显示所选' if all_hidden else '隐藏所选')
        self.hide_selected_btn.setEnabled(bool(indices) and not self.task)
        self.restore_points_btn.setEnabled(bool(hidden) and not self.task)
        for widget in (self.confirm_btn,self.delete_btn,self.event_edit_btn,self.event_point_btn):widget.setEnabled(row is not None and not self.task)
        confirmed=bool(indices) and all(self.project['events'][i]['confirmed'] for i in indices)
        self.confirm_btn.setText(('撤销所选确认' if confirmed else '确认所选') if len(indices)>1 else ('撤销确认' if confirmed else '确认当前'))
        self.delete_btn.setText('删除所选' if len(indices)>1 else '删除当前')
        self.table.update()

    def update_map_count(self):
        total=sum(event_kind(e)=='bounce' and bool(e.get('table_xy')) for e in self.table.events)
        self.map_count.setText(f'落点 {len(self.table.visible_events())}/{total} · 隐藏 {len(self.table.hidden)}项')
        self.map_count.setToolTip(f'事件列表筛选 {len(self.visible_rows)} 项；隐藏和筛选均不会删除标注。')

    def set_visibility(self,numbers,hidden):
        if not self.project or self.task:return
        current=hidden_numbers(self.project)
        current.update(numbers) if hidden else current.difference_update(numbers)
        self.project.setdefault('view_settings',{})['hidden_event_numbers']=sorted(current)
        self.dirty=True;self.refresh_events();self.refresh_frame()
        self.status.setText('已更新显示：隐藏不删除标签，也不改变审核状态；保存项目可保留此设置')

    def toggle_selected_visibility(self):
        indices=self.selected_indices()
        if not indices:return
        numbers=[self.project['events'][i]['event_number'] for i in indices]
        self.set_visibility(numbers,not all(n in hidden_numbers(self.project) for n in numbers))

    def hide_number(self,number):self.set_visibility([number],True)

    def restore_all_points(self):
        if self.project:self.set_visibility(list(hidden_numbers(self.project)),False)

    def toggle_numbers(self,checked):
        self.table.numbers=checked;self.table.prepare_layout();self.table.update();self.refresh_frame()

    def set_selected_confirmation(self,confirmed):
        indices=self.selected_indices()
        if not indices or self.task:return
        self.remember_annotation()
        for i in indices:
            e=self.project['events'][i];before=copy.deepcopy(e)
            e.update(confirmed=confirmed,human_reviewed=confirmed,type=event_kind(e) if confirmed else event_kind(e)+'_candidate')
            log_edit(self.project,'event_confirmation',before,e)
        self.dirty=True;self.refresh_events();self.refresh_frame()

    def event_context_menu(self,position):
        item=self.events.itemAt(position)
        if item is None:return
        if item.row() not in {i.row() for i in self.events.selectionModel().selectedRows()}:self.events.selectRow(item.row())
        menu=QMenu(self)
        menu.addAction('隐藏所选',lambda:self.set_visibility([self.project['events'][i]['event_number'] for i in self.selected_indices()],True))
        menu.addAction('显示所选',lambda:self.set_visibility([self.project['events'][i]['event_number'] for i in self.selected_indices()],False))
        menu.addAction('确认所选',lambda:self.set_selected_confirmation(True))
        menu.addAction('撤销所选确认',lambda:self.set_selected_confirmation(False))
        menu.addSeparator();menu.addAction('编辑当前事件',self.edit_selected_event)
        menu.addAction('复制当前详情',lambda:QApplication.clipboard().setText(self.selection_details.text()))
        menu.exec(self.events.viewport().mapToGlobal(position))

    def refresh_events(self):
        prior_selection={int(self.events.item(item.row(),0).text().lstrip('#')) for item in self.events.selectionModel().selectedRows() if self.events.item(item.row(),0)}
        events=self.project['events'] if self.project else []
        if self.project:ensure_numbers(self.project)
        if self.project and 'view_settings' in self.project:
            valid={e['event_number'] for e in events}
            self.project['view_settings']['hidden_event_numbers']=sorted(hidden_numbers(self.project)&valid)
        self.visible_rows=self.filtered_indices();hidden=hidden_numbers(self.project or {})
        self.events.blockSignals(True);self.events.setRowCount(len(self.visible_rows))
        for row,index in enumerate(self.visible_rows):
            e=events[index]
            kind=event_kind(e);xy=e.get('table_xy');name={'bounce':'触台','out':'出界','net':'拦网','stroke':'击球'}.get(kind,kind)
            status='已确认' if e['confirmed'] else ('待复核·补检' if e.get('review_priority')=='high' else '待复核')
            values=[f"#{e['event_number']:03d}",str(e['frame']),f"{e['time']:.3f}",name if e['confirmed'] or kind=='bounce' else '疑似'+name,
                f'{xy[0]:.2f}, {xy[1]:.2f}' if xy else '—',status,'隐藏' if e['event_number'] in hidden else '显示']
            for col,value in enumerate(values):
                item=QTableWidgetItem(value);item.setToolTip(f"#{e['event_number']:03d} · 帧{e['frame']} · {value}")
                item.setForeground(QColor('#6f829d' if e['event_number'] in hidden else ('#6de0a3' if e['confirmed'] else '#ffba66')));self.events.setItem(row,col,item)
                if 'fit_frame' in e:item.setToolTip(f"拟合反弹时刻约为第 {e['fit_frame']:.1f} 帧；落点仍需人工复核。")
                if e.get('edge_uncertain'):item.setToolTip('靠近边线：请区分擦边触台与出界，必要时修正边界或事件类型。')
        selected=next((row for row,index in enumerate(self.visible_rows) if events[index]['event_number']==self.selected_number),None)
        if selected is not None:self.events.setCurrentCell(selected,0);self.events.selectRow(selected)
        else:self.events.clearSelection();self.events.setCurrentCell(-1,-1)
        for row,index in enumerate(self.visible_rows):
            if events[index]['event_number'] in prior_selection:
                self.events.selectionModel().select(self.events.model().index(row,0),QItemSelectionModel.Select|QItemSelectionModel.Rows)
        self.events.blockSignals(False)
        self.table.events=events;self.table.hidden=hidden;self.table.allowed={events[i]['event_number'] for i in self.visible_rows};self.table.update()
        self.table.prepare_layout()
        self.update_map_count()
        self.update_selection_details()
        self.hints_list.clear()
        if self.project:
            count=sum(p['x'] is not None for p in self.project['tracks']);confirmed=sum(e['confirmed'] for e in events)
            speed=self.project['performance']['end_to_end_fps']
            backend='显卡' if self.project['performance']['provider']=='DmlExecutionProvider' else 'CPU'
            near=sum(event_kind(e)=='bounce' and table_half(e)=='near' for e in events)
            far=sum(event_kind(e)=='bounce' and table_half(e)=='far' for e in events)
            net=sum(event_kind(e)=='net' for e in events);out=sum(event_kind(e)=='out' for e in events)
            reviewed=self.project.get('review',{}).get('status')=='approved'
            self.stats.setText(f'球心帧  {count} / {self.project["frame_count"]}\n触台 近端 {near} / 远端 {far}\n出界 {out} / 拦网 {net}\n待复核 {len(events)-confirmed} / 已确认 {confirmed}\n审核状态 {"已入库" if reviewed else "待审核"}\n处理 {speed:.1f} FPS（{backend}）')
            hints=rally_hints(events,self.fps,self.project.get('break_frames',[]));self.project['review_hints']=hints
            for h in hints:
                self.hints_list.addItem(f"帧 {h['start']}–{h['end']}：检查另一侧漏点")
                self.hints_list.item(self.hints_list.count()-1).setData(Qt.UserRole,h)
            if not hints:self.hints_list.addItem('暂无同侧连续触台提示；数量不强制相等')
        else:self.stats.setText('检测帧  —\n候选触台  —\n人工确认  —\n处理速度  —')

    def event_clicked(self,row,col):
        if self.project and 0<=row<len(self.visible_rows):
            self.update_selection_details();self.pause();self.seek(self.project['events'][self.visible_rows[row]]['frame']);self.refresh_frame()

    def selected_event(self):
        row=self.events.currentRow()
        return self.visible_rows[row] if self.project and 0<=row<len(self.visible_rows) else None

    def confirm_event(self):
        indices=self.selected_indices()
        if len(indices)>1:
            self.set_selected_confirmation(not all(self.project['events'][i]['confirmed'] for i in indices));return
        row=self.selected_event()
        if row is None:return
        self.remember_annotation()
        e=self.project['events'][row];before=copy.deepcopy(e);e['confirmed']=not e['confirmed'];e['type']=event_kind(e) if e['confirmed'] else event_kind(e)+'_candidate'
        e['human_reviewed']=e['confirmed'];log_edit(self.project,'event_confirmation',before,e)
        self.dirty=True;self.refresh_events()
        visible=[self.project['events'][i] for i in self.visible_rows]
        target=next((p for p in visible if p['event_number']==e['event_number']),None)
        if target is None:target=next((p for p in visible if p['frame']>e['frame']),visible[0] if visible else None)
        if target:self.select_number(target['event_number'])
        self.refresh_frame()

    def delete_event(self):
        indices=self.selected_indices()
        if indices:
            self.remember_annotation()
            for row in reversed(indices):
                before=self.project['events'].pop(row);self.project.setdefault('rejected_events',[]).append(copy.deepcopy(before))
                log_edit(self.project,'event_rejected',before,None)
            self.dirty=True;self.refresh_events();self.refresh_frame()

    def add_event(self):
        if not self.project:return
        self.pan_button.setChecked(False)
        self.pause();self.pending_add_event=True;self.editing=False;self.edit_event_ref=None;self.calibrating=False
        self.status.setText('点击实际事件位置，再选择触台、出界或拦网类型');self.refresh_frame()

    def change_heat(self,checked):self.table.heat=checked;self.table.update()

    def save(self):
        if not self.project:return True
        path,_=QFileDialog.getSaveFileName(self,'保存分析项目',str(Path(self.video).with_suffix('.analysis.json')),'分析项目 (*.json)')
        if not path:return False
        try:save_project(path,self.project);self.dirty=False;self.status.setText('分析项目已保存');return True
        except Exception as e:self.error(str(e));return False

    def export_data(self):
        if not self.project:return
        folder=QFileDialog.getExistingDirectory(self,'选择导出文件夹')
        if folder:
            existing=[x for x in ('tracks.csv','events.csv','project.json') if (Path(folder)/x).exists()]
            if existing and QMessageBox.question(self,'覆盖导出文件','该目录已有分析文件，是否覆盖？')!=QMessageBox.Yes:return
            try:export_csv(folder,self.project);self.status.setText('已导出 tracks.csv、events.csv 和 project.json')
            except Exception as e:self.error(str(e))

    def export_movie(self):
        if not self.project:return
        self.pause()
        path,_=QFileDialog.getSaveFileName(self,'导出轨迹视频（不含音频）',str(Path(self.video).with_name(Path(self.video).stem+'_tracked.mp4')),'视频 (*.mp4)')
        if path:self.launch_task('export',dict(path=path,project=self.project,render_options=self.render_options()))

    def export_distribution_ui(self,synchronized):
        if not self.project or self.task:return
        if len(self.corners)!=4:self.error('请先标定球台四角，再导出二维落点。');return
        self.pause();dialog=DistributionExportDialog(synchronized,self)
        if dialog.exec()!=QDialog.Accepted:return
        suffix='_轨迹与落点.mp4' if synchronized else '_落点分布.png'
        path,_=QFileDialog.getSaveFileName(self,'保存同步视频' if synchronized else '保存二维落点图',str(Path(self.video).with_name(Path(self.video).stem+suffix)),
            '视频 (*.mp4)' if synchronized else 'PNG图片 (*.png)')
        if not path:return
        options={key:value for key,value in self.render_options().items() if key in ('hidden_event_numbers','visible_event_numbers','event_numbers')} if dialog.current_view.isChecked() else {}
        args=dict(path=path,project=copy.deepcopy(self.project),options=options,confirmed_only=dialog.confirmed.isChecked())
        if synchronized:args['height']=dialog.height.currentData()
        self.launch_task('sync' if synchronized else 'plot',args)

    def update_controls(self):
        busy=self.task is not None;has=self.video is not None;analyzed=self.project is not None
        for widget in (self.open_btn,self.project_btn,self.model_btn,self.model_preset,self.threshold,self.acceleration,self.auto_on_load,self.motion_filter):widget.setEnabled(not busy)
        for widget in (self.cal_btn,self.auto_cal_btn,self.rotate_cal_btn,self.clear_cal_btn,self.run_btn,self.play_btn,self.back_btn,self.next_btn,self.slider,self.playback_speed,self.step_input,self.goto_input,self.goto_btn):widget.setEnabled(has and not busy)
        for widget in (self.range_start_btn,self.range_end_btn):widget.setEnabled(has and not busy)
        for widget in (self.save_btn,self.csv_btn,self.export_btn,self.edit_btn,self.add_btn,self.confirm_btn,self.delete_btn,self.recalc_btn,
            self.event_edit_btn,self.event_point_btn,self.hide_ball_btn,self.approve_btn,self.plot_btn,self.sync_btn):widget.setEnabled(analyzed and not busy)
        self.training_btn.setEnabled(not busy)
        for widget in (self.prev_event_btn,self.next_event_btn):widget.setEnabled(analyzed and not busy)
        self.undo_btn.setEnabled(analyzed and not busy and bool(self.history.items))
        self.update_selection_details()
        self.cancel_btn.setEnabled(busy)
        if not busy or not self.task.cancel.is_set():self.cancel_btn.setText('取消任务')

    def closeEvent(self,event):
        if self.live_panel.worker:
            self.pending_close=True;self.live_panel.stop_session();event.ignore();return
        if self.task:
            self.pending_close=True;self.cancel_task();event.ignore();return
        if not self.safe_to_replace():self.pending_close=False;event.ignore();return
        self.pause()
        if self.cap:self.cap.release()
        event.accept()


def main():
    app=QApplication(sys.argv);app.setApplicationName('PingPongStudio')
    configure_app(app)
    cleanup_legacy_aliases()
    approval_root().mkdir(parents=True,exist_ok=True)
    window=MainWindow()
    screen=app.primaryScreen().availableGeometry()
    window.resize(min(1440,screen.width()-40),min(920,screen.height()-50))
    window.show()
    if len(sys.argv)>1 and Path(sys.argv[1]).is_file():
        if window.load_video(sys.argv[1]):QTimer.singleShot(0,window.auto_calibrate)
    else:
        sample=asset_path('sample/demo.mp4')
        if sample.exists() and window.load_video(sample):QTimer.singleShot(0,window.auto_calibrate)
    return app.exec()
