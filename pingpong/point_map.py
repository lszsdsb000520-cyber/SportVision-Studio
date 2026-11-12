"""Interactive, numbered table map with overlap cycling and display-only hiding."""
from PySide6.QtCore import Qt,QRectF,QPointF,Signal
from PySide6.QtGui import QPainter,QPen,QColor,QFont
from PySide6.QtWidgets import QWidget,QToolTip,QMenu
from .events import event_kind
from .label_layout import label_slots
from math import ceil


class TableView(QWidget):
    event_selected=Signal(int)
    hide_requested=Signal(int)
    restore_requested=Signal()

    def __init__(self):
        super().__init__();self.events=[];self.heat=False;self.show_points=True
        self.hidden=set();self.allowed=None;self.selected=None;self.numbers=True
        self.label_rects={}
        self.last_hits=();self.cycle=0;self.last_click=QPointF(-100,-100)
        self.setMinimumSize(205,280);self.setMouseTracking(True);self.setCursor(Qt.PointingHandCursor)

    def table_rect(self):
        available=QRectF(46,30,self.width()-92,self.height()-62)
        width=min(available.width(),available.height()*1.525/2.74)
        height=width*2.74/1.525
        return QRectF((self.width()-width)/2,(self.height()-height)/2,width,height)

    def point_position(self,event):
        r=self.table_rect();xy=event['table_xy']
        return QPointF(r.left()+xy[0]/1.525*r.width(),r.top()+xy[1]/2.74*r.height())

    def visible_events(self):
        if not self.show_points:return []
        return [e for e in self.events if event_kind(e)=='bounce' and e.get('table_xy')
                and e.get('event_number') not in self.hidden
                and (self.allowed is None or e.get('event_number') in self.allowed)]

    def prepare_layout(self):
        needed=max(280,ceil(len(self.visible_events())/2)*20+70) if self.numbers else 280
        if self.minimumHeight()!=needed:self.setMinimumHeight(needed)

    def hits(self,position):
        visible=self.visible_events()
        labels=[e for e in visible if e.get('event_number') in self.label_rects and QRectF(*self.label_rects[e['event_number']]).contains(position)]
        if labels:return labels
        return sorted([e for e in visible if (self.point_position(e)-position).manhattanLength()<18],
                      key=lambda e:((self.point_position(e)-position).manhattanLength(),e.get('event_number',0)))

    def mousePressEvent(self,event):
        hits=self.hits(event.position())
        if event.button()==Qt.RightButton:
            menu=QMenu(self)
            for target in hits[:12]:
                number=target['event_number']
                menu.addAction(f"查看 #{number:03d} · 第{target['frame']}帧",lambda checked=False,n=number:self.event_selected.emit(n))
                menu.addAction(f'隐藏 #{number:03d}',lambda checked=False,n=number:self.hide_requested.emit(n))
            menu.addSeparator();menu.addAction('恢复全部隐藏落点',self.restore_requested.emit)
            menu.exec(event.globalPosition().toPoint());return
        if event.button()!=Qt.LeftButton or not hits:return
        identifiers=tuple(e['event_number'] for e in hits)
        repeated=identifiers==self.last_hits and (event.position()-self.last_click).manhattanLength()<5
        self.cycle=(self.cycle+1)%len(hits) if repeated else 0
        self.last_hits=identifiers;self.last_click=event.position()
        self.event_selected.emit(hits[self.cycle]['event_number'])

    def mouseMoveEvent(self,event):
        hits=self.hits(event.position())
        if not hits:QToolTip.hideText();return
        lines=[f"#{e['event_number']:03d} · 帧 {e['frame']} · {e['time']:.3f}秒 · "+('已确认' if e['confirmed'] else '待复核') for e in hits[:6]]
        if len(hits)>1:lines.append('重复点击可切换重叠落点；右键可单独隐藏')
        QToolTip.showText(event.globalPosition().toPoint(),'\n'.join(lines),self)

    def paintEvent(self,event):
        p=QPainter(self);p.setRenderHint(QPainter.Antialiasing);p.fillRect(self.rect(),QColor('#111d30'))
        r=self.table_rect();p.fillRect(r,QColor('#133f4a'));p.setPen(QPen(QColor('#b2d9d6'),2));p.drawRect(r)
        p.drawLine(QPointF(r.center().x(),r.top()),QPointF(r.center().x(),r.bottom()))
        p.setPen(QPen(QColor('#e4f3f2'),3));p.drawLine(QPointF(r.left()-7,r.center().y()),QPointF(r.right()+7,r.center().y()))
        events=sorted(self.visible_events(),key=lambda e:e.get('event_number')==self.selected)
        labelled=events if self.numbers else [e for e in events if e.get('event_number')==self.selected]
        p.setFont(QFont('Microsoft YaHei UI',9,QFont.Bold))
        label_width=max([38]+[p.fontMetrics().horizontalAdvance(f"#{e['event_number']:03d}")+4 for e in labelled])
        self.label_rects=label_slots([(e['event_number'],self.point_position(e).x(),self.point_position(e).y()) for e in labelled],self.width(),self.height(),label_width)
        for e in events:
            point=self.point_position(e);color=QColor('#6de0a3' if e['confirmed'] else '#ffba66')
            if self.heat:
                for radius in range(20,1,-2):
                    color.setAlpha(12);p.setBrush(color);p.setPen(Qt.NoPen);p.drawEllipse(point,radius,radius)
            color.setAlpha(255);p.setBrush(color);p.setPen(Qt.NoPen);p.drawEllipse(point,5,5)
            selected=e.get('event_number')==self.selected
            if selected:
                p.setBrush(Qt.NoBrush);p.setPen(QPen(QColor('#ffffff'),2));p.drawEllipse(point,10,10)
            if self.numbers or selected:
                p.setFont(QFont('Microsoft YaHei UI',9,QFont.Bold));p.setPen(QColor('#ffffff' if selected else '#dbe9f7'))
                text=f"#{e.get('event_number',0):03d}"
                rectangle=QRectF(*self.label_rects[e['event_number']])
                if rectangle is not None:
                    anchor=QPointF(rectangle.right() if rectangle.center().x()<r.center().x() else rectangle.left(),rectangle.center().y())
                    p.setPen(QPen(QColor('#ffffff' if selected else '#657f90'),2 if selected else 1));p.drawLine(point,anchor)
                    p.fillRect(rectangle,QColor('#173044'));p.setPen(QColor('#ffffff' if selected else '#dbe9f7'))
                    p.drawText(rectangle,Qt.AlignCenter,text)
        p.setPen(QColor('#91a4b9'));p.setFont(QFont('Microsoft YaHei UI',9))
        p.drawText(QRectF(0,0,self.width(),24),Qt.AlignCenter,'远端')
        p.drawText(QRectF(0,self.height()-24,self.width(),24),Qt.AlignCenter,'近端 · 点击落点联动回看')
