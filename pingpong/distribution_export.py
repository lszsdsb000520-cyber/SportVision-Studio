"""Final table plots and frame-synchronous video+table exports."""
from pathlib import Path
import os
import cv2,numpy as np
from PySide6.QtCore import Qt,QRectF,QPointF
from PySide6.QtGui import QImage,QPainter,QColor,QPen,QFont,QFontDatabase
from PySide6.QtWidgets import QApplication
from .events import event_kind
from .detector import render_frame
from .cancellation import checkpoint,cancellable_result
from .label_layout import label_slots
from math import ceil
from .video_io import VideoReader,VideoWriter

_app=None


def ensure_app():
    global _app
    if QApplication.instance() is None:
        os.environ.setdefault('QT_QPA_PLATFORM','offscreen');_app=QApplication([])
    app=QApplication.instance()
    if not app.property('pingpong_theme_ready') and not app.property('pingpong_export_fonts_ready'):
        for name in ('msyh.ttc','msyhbd.ttc'):
            path=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts'/name
            if path.exists():QFontDatabase.addApplicationFont(str(path))
        app.setFont(QFont('Microsoft YaHei UI',10));app.setProperty('pingpong_export_fonts_ready',True)


def visible_bounces(project,options=None,index=None,confirmed_only=False):
    options=options or {};allowed=options.get('visible_event_numbers');hidden=options.get('hidden_event_numbers',[])
    return [e for e in project['events'] if event_kind(e)=='bounce' and e.get('table_xy')
        and (index is None or e['frame']<=index) and (not confirmed_only or e['confirmed'])
        and e.get('event_number') not in hidden and (allowed is None or e.get('event_number') in allowed)]


def distribution_image(project,width=640,height=960,index=None,options=None,confirmed_only=False,dark=False):
    ensure_app();options=options or {};events=visible_bounces(project,options,index,confirmed_only)
    if index is None:height=max(height,ceil(len(events)/2)*22+230)
    image=QImage(width,height,QImage.Format_RGB32)
    image.fill(QColor('#101d2e' if dark else '#ffffff'))
    painter=QPainter(image);painter.setRenderHint(QPainter.Antialiasing)
    text=QColor('#e8f0f8' if dark else '#172b42');muted=QColor('#92a7bd' if dark else '#61758a')
    painter.setPen(text);painter.setFont(QFont('Microsoft YaHei UI',max(11,width//30),QFont.Bold))
    painter.drawText(QRectF(12,12,width-24,34),Qt.AlignCenter,'乒乓球落点分布')
    painter.setFont(QFont('Microsoft YaHei UI',max(9,width//48)));painter.setPen(muted)
    timestamp=project['tracks'][index]['time'] if index is not None else None
    title=f'第 {index} 帧 · {timestamp:.3f} 秒' if index is not None else Path(project['video']).name
    painter.drawText(QRectF(12,50,width-24,28),Qt.AlignCenter,title)
    available_height=max(80,height-230);court_width=min(width-100,available_height*1.525/2.74);court_height=court_width*2.74/1.525
    court=QRectF((width-court_width)/2,105,court_width,court_height)
    painter.fillRect(court,QColor('#163f49' if dark else '#edf6f7'))
    painter.setPen(QPen(QColor('#bad4df' if dark else '#3a6679'),2));painter.drawRect(court)
    painter.drawLine(QPointF(court.center().x(),court.top()),QPointF(court.center().x(),court.bottom()))
    painter.setPen(QPen(QColor('#e7f4fa' if dark else '#36536b'),3))
    painter.drawLine(QPointF(court.left()-8,court.center().y()),QPointF(court.right()+8,court.center().y()))
    painter.setPen(muted)
    painter.drawText(QRectF(court.left(),80,court.width(),22),Qt.AlignCenter,'远端')
    painter.drawText(QRectF(court.left(),court.bottom()+5,court.width(),22),Qt.AlignCenter,'近端')
    label_font=max(6,min(max(9,width//48),int((height-230)/max(1,ceil(len(events)/2))*.65)))
    painter.setFont(QFont('Microsoft YaHei UI',label_font))
    labels={e.get('event_number',project['events'].index(e)+1):e for e in events}
    label_width=max([32]+[painter.fontMetrics().horizontalAdvance(f'#{n:03d}')+4 for n in labels])
    slots=label_slots([(n,court.left()+e['table_xy'][0]/1.525*court.width(),court.top()+e['table_xy'][1]/2.74*court.height()) for n,e in labels.items()],
                     width,height,label_width,22,95,110)
    for e in (list(reversed(events)) if index is not None else events):
        xy=e['table_xy'];point=QPointF(court.left()+xy[0]/1.525*court.width(),court.top()+xy[1]/2.74*court.height())
        color=QColor('#33c49f' if e['confirmed'] else '#ed9c32')
        painter.setPen(QPen(color,2));painter.setBrush(color if e['confirmed'] else Qt.NoBrush);painter.drawEllipse(point,5,5)
        if index is not None and timestamp-e['time']<=.3:
            painter.setBrush(Qt.NoBrush);painter.setPen(QPen(text,2));painter.drawEllipse(point,11,11)
        if options.get('event_numbers',True):
            number=e.get('event_number',project['events'].index(e)+1);rect=QRectF(*slots[number])
            anchor=QPointF(rect.right() if rect.center().x()<court.center().x() else rect.left(),rect.center().y())
            painter.setPen(QPen(muted,1));painter.drawLine(point,anchor)
            painter.fillRect(rect,QColor('#163441' if dark else '#edf6f7'));painter.setPen(text);painter.drawText(rect,Qt.AlignCenter,f'#{number:03d}')
    painter.setFont(QFont('Microsoft YaHei UI',max(9,width//48)))
    confirmed=sum(e['confirmed'] for e in events);bottom=court.bottom()+35
    painter.setPen(text);painter.drawText(QRectF(10,bottom,width-20,24),Qt.AlignCenter,f'已确认 {confirmed} · 候选 {len(events)-confirmed} · 合计 {len(events)}')
    painter.setPen(muted);painter.drawText(QRectF(10,bottom+27,width-20,22),Qt.AlignCenter,'绿：已确认   橙：待复核   单位：米')
    painter.drawText(QRectF(10,height-30,width-20,24),Qt.AlignCenter,'按视频触台时间同步显示' if index is not None else '台面尺寸 1.525 × 2.740 米')
    painter.end();return image


def check_corners(project):
    if len(project.get('corners',[]))!=4:raise ValueError('请先标定球台四角，才能导出二维落点。')


@cancellable_result
def export_distribution(path,project,options=None,confirmed_only=False,cancelled=None):
    checkpoint(cancelled)
    check_corners(project);path=Path(path)
    if path.resolve()==Path(project['video']).resolve():raise ValueError('导出不能覆盖原视频。')
    image=distribution_image(project,options=options,confirmed_only=confirmed_only)
    checkpoint(cancelled)
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+'.partial.png')
    if temporary.exists():raise FileExistsError('临时图片已存在，请选择其他名称。')
    try:
        if not image.save(str(temporary),'PNG'):raise IOError('无法写入落点图。')
        checkpoint(cancelled)
        temporary.replace(path)
    finally:
        if temporary.exists():temporary.unlink()
    return str(path)


@cancellable_result
def export_synchronized(path,project,options=None,confirmed_only=False,height=720,progress=None,cancelled=None):
    checkpoint(cancelled)
    check_corners(project);path=Path(path)
    if path.resolve()==Path(project['video']).resolve():raise ValueError('导出不能覆盖原视频。')
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.stem+'.partial.mp4')
    if temporary.exists():raise FileExistsError('临时视频已存在，请选择其他名称。')
    height=int(height)//2*2;left_width=round(project['width']/project['height']*height/2)*2;panel_width=400 if height<=720 else 560
    cap=VideoReader(project['video'],cancelled);writer=VideoWriter(temporary,project['fps'],(left_width+panel_width,height))
    complete=False;options=dict(options or {});options.pop('selected_event_number',None)
    if confirmed_only:
        allowed=options.get('visible_event_numbers')
        options['visible_event_numbers']=[e.get('event_number') for e in project['events'] if e['confirmed'] and (allowed is None or e.get('event_number') in allowed)]
    try:
        if not cap.isOpened() or not writer.isOpened():raise IOError('无法打开输入或创建同步视频。')
        for index in range(project['frame_count']):
            if cancelled and cancelled():return None
            ok,frame=cap.read()
            if not ok:raise IOError(f'第{index}帧解码失败，导出已取消。')
            annotated=render_frame(frame,project,index,options=options)
            left=cv2.resize(annotated,(left_width,height));panel=distribution_image(project,panel_width,height,index,options,confirmed_only,True)
            panel=panel.convertToFormat(QImage.Format_BGR888)
            right=np.frombuffer(panel.bits(),np.uint8).reshape(height,panel.bytesPerLine())[:,:panel_width*3].reshape(height,panel_width,3).copy()
            writer.write(np.concatenate([left,right],axis=1),project['tracks'][index]['time'])
            if progress and index%15==0:progress(index+1,project['frame_count'],None,None)
        checkpoint(cancelled);complete=True
    finally:
        cap.release();writer.release()
        if not complete and temporary.exists():temporary.unlink()
    temporary.replace(path);return str(path)
