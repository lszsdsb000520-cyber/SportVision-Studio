"""Offscreen interaction checks for original-image coordinates after zoom/pan."""
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PySide6.QtCore import QPointF,Qt
from PySide6.QtGui import QMouseEvent,QWheelEvent
from PySide6.QtWidgets import QApplication
from pingpong.ui import VideoView


def mouse(view,kind,position,button,buttons):
    event=QMouseEvent(kind,position,position,button,buttons,Qt.NoModifier)
    QApplication.sendEvent(view,event)


def main():
    from PySide6.QtCore import QEvent,QPoint
    app=QApplication([]);view=VideoView();view.resize(800,500)
    view.set_frame(np.zeros((1080,1920,3),np.uint8));view.show();app.processEvents()
    clicks=[];corners=[];view.clicked.connect(lambda x,y:clicks.append((x,y)))
    view.corner_moved.connect(lambda *values:corners.append(values))
    anchor=QPointF(350,220);old=view.image_rect()
    uv=((anchor.x()-old.x())/old.width(),(anchor.y()-old.y())/old.height())
    wheel=QWheelEvent(anchor,anchor,QPoint(),QPoint(0,120),Qt.NoButton,Qt.NoModifier,Qt.NoScrollPhase,False)
    QApplication.sendEvent(view,wheel);assert abs(view.zoom-1.2)<1e-6
    new=view.image_rect()
    assert np.allclose(uv,[(anchor.x()-new.x())/new.width(),(anchor.y()-new.y())/new.height()])
    view.zoom_at(3)
    mouse(view,QEvent.MouseButtonPress,QPointF(400,250),Qt.RightButton,Qt.RightButton)
    mouse(view,QEvent.MouseMove,QPointF(440,270),Qt.NoButton,Qt.RightButton)
    mouse(view,QEvent.MouseButtonRelease,QPointF(440,270),Qt.RightButton,Qt.NoButton)
    assert not clicks and view.pan_start is None
    point=QPointF(420,240);rect=view.image_rect()
    expected=((point.x()-rect.x())/rect.width()*1920,(point.y()-rect.y())/rect.height()*1080)
    mouse(view,QEvent.MouseButtonPress,point,Qt.LeftButton,Qt.LeftButton)
    mouse(view,QEvent.MouseButtonRelease,point,Qt.LeftButton,Qt.NoButton)
    assert np.allclose(clicks[-1],expected)
    view.allow_drag=True;view.corners=[expected,(0,0),(1919,0),(1919,1079)]
    mouse(view,QEvent.MouseButtonPress,point,Qt.LeftButton,Qt.LeftButton)
    target=point+QPointF(15,9)
    mouse(view,QEvent.MouseMove,target,Qt.NoButton,Qt.LeftButton)
    mouse(view,QEvent.MouseButtonRelease,target,Qt.LeftButton,Qt.NoButton)
    assert corners[-1][0]==0 and corners[-1][-1] is True
    assert np.allclose(corners[-1][1:3],[(target.x()-rect.x())/rect.width()*1920,(target.y()-rect.y())/rect.height()*1080])
    zoom=view.zoom;view.set_frame(np.ones((1080,1920,3),np.uint8));assert view.zoom==zoom
    view.reset_zoom();assert view.zoom==1 and view.pan==QPointF()
    view.zoom_at(100);assert view.zoom==8
    view.zoom_at(.001);assert view.zoom==1
    print('PASS: wheel anchor, pan, original-coordinate click/corner drag, frame persistence, reset and limits')
    view.close()


if __name__=='__main__':main()
