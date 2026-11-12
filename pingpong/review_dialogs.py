from PySide6.QtWidgets import (QDialog,QVBoxLayout,QFormLayout,QSpinBox,QDoubleSpinBox,
    QComboBox,QCheckBox,QDialogButtonBox,QPushButton,QLabel)
from .events import event_kind


class InvisibleRangeDialog(QDialog):
    def __init__(self,total,current,start,end,parent=None):
        super().__init__(parent);self.setWindowTitle('批量标注球不可见');self.resize(470,270)
        layout=QVBoxLayout(self);form=QFormLayout();layout.addLayout(form)
        self.start=QSpinBox();self.start.setRange(0,total-1);self.start.setValue(start)
        self.end=QSpinBox();self.end.setRange(0,total-1);self.end.setValue(end)
        form.addRow('起始帧（包含）',self.start);form.addRow('结束帧（包含）',self.end)
        one=QPushButton('只标当前帧')
        one.clicked.connect(lambda:(self.start.setValue(current),self.end.setValue(current)));layout.addWidget(one)
        self.summary=QLabel();layout.addWidget(self.summary)
        self.remove_events=QCheckBox('同时删除区间内事件（含已确认事件，可撤销）');layout.addWidget(self.remove_events)
        note=QLabel('默认保留事件：小球被遮挡时仍可能触台。区间内球心会记为人工确认不可见，不能仅因模型漏检就批量标记。')
        note.setWordWrap(True);layout.addWidget(note)
        buttons=QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText('应用不可见标注');buttons.button(QDialogButtonBox.Cancel).setText('取消')
        def refresh():
            valid=self.start.value()<=self.end.value()
            self.summary.setText(f'共 {self.end.value()-self.start.value()+1} 帧' if valid else '起始帧不能大于结束帧')
            buttons.button(QDialogButtonBox.Ok).setEnabled(valid)
        self.start.valueChanged.connect(refresh);self.end.valueChanged.connect(refresh);refresh()
        buttons.accepted.connect(self.accept);buttons.rejected.connect(self.reject);layout.addWidget(buttons)


class EventDialog(QDialog):
    def __init__(self,event,project,current_frame,parent=None):
        super().__init__(parent);self.setWindowTitle('编辑人工事件');self.resize(410,340)
        layout=QVBoxLayout(self);form=QFormLayout();layout.addLayout(form)
        self.kind=QComboBox()
        for text,value in [('触台落点','bounce'),('击球接触','stroke'),('出界','out'),('被网拦下','net')]:self.kind.addItem(text,value)
        self.kind.setCurrentIndex(max(0,self.kind.findData(event_kind(event))));form.addRow('事件类型',self.kind)
        self.frame=QSpinBox();self.frame.setRange(0,project['frame_count']-1);self.frame.setValue(event['frame']);form.addRow('事件帧号',self.frame)
        self.x=QDoubleSpinBox();self.x.setRange(0,project['width']-1);self.x.setDecimals(2);self.x.setValue(event['x']);form.addRow('画面 X',self.x)
        self.y=QDoubleSpinBox();self.y.setRange(0,project['height']-1);self.y.setDecimals(2);self.y.setValue(event['y']);form.addRow('画面 Y',self.y)
        take=QPushButton('使用当前回放帧和球心')
        def use_current():
            self.frame.setValue(current_frame);point=project['tracks'][current_frame]
            if point['x'] is not None:self.x.setValue(point['x']);self.y.setValue(point['y'])
        take.clicked.connect(use_current);layout.addWidget(take)
        self.confirmed=QCheckBox('我已确认这个事件');self.confirmed.setChecked(True);layout.addWidget(self.confirmed)
        buttons=QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText('保存修正');buttons.button(QDialogButtonBox.Cancel).setText('取消')
        buttons.accepted.connect(self.accept);buttons.rejected.connect(self.reject);layout.addWidget(buttons)

    def values(self):
        return dict(kind=self.kind.currentData(),frame=self.frame.value(),x=self.x.value(),y=self.y.value(),confirmed=self.confirmed.isChecked())


class DistributionExportDialog(QDialog):
    def __init__(self,synchronized,parent=None):
        super().__init__(parent);self.setWindowTitle('导出轨迹与同步落点视频' if synchronized else '导出二维落点分布图')
        layout=QVBoxLayout(self)
        label=QLabel('同步视频左侧显示轨迹，右侧按触台帧累计落点；帧率与原视频一致，不含音频。' if synchronized else '导出最终二维落点分布PNG，编号对应事件列表。候选和已确认落点使用不同颜色。')
        label.setWordWrap(True);layout.addWidget(label)
        self.current_view=QCheckBox('采用当前隐藏与事件筛选');self.current_view.setChecked(True);layout.addWidget(self.current_view)
        self.confirmed=QCheckBox('仅显示已确认事件');layout.addWidget(self.confirmed)
        self.height=QComboBox();self.height.addItem('720p（推荐）',720);self.height.addItem('1080p（更清晰）',1080)
        if synchronized:layout.addWidget(self.height)
        buttons=QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText('选择保存位置');buttons.button(QDialogButtonBox.Cancel).setText('取消')
        buttons.accepted.connect(self.accept);buttons.rejected.connect(self.reject);layout.addWidget(buttons)


class ApprovalDialog(QDialog):
    def __init__(self,project,parent=None):
        super().__init__(parent);self.setWindowTitle('人工审核通过并入库');self.resize(530,290)
        layout=QVBoxLayout(self)
        text=QLabel('将保存原视频、修正后的事件、球心标注及审核范围。剩余事件会标为已人工确认；请先补齐漏检并删除误检。每次入库创建新版本。')
        text.setWordWrap(True);layout.addWidget(text)
        hidden=set(project.get('view_settings',{}).get('hidden_event_numbers',[]))
        count=sum(e.get('event_number') in hidden for e in project['events'])
        if count:
            note=QLabel(f'当前有 {count} 项事件被隐藏，它们仍会保存并进入审核数据。隐藏只影响显示；误检请删除后再审核。')
            note.setWordWrap(True);note.setStyleSheet('color:#ffba66');layout.addWidget(note)
        self.events=QCheckBox('我已复核整段的触台、出界和拦网事件，以及球台边界');layout.addWidget(self.events)
        self.ball=QCheckBox('我还逐帧核对了全部球心及不可见帧（可选）');layout.addWidget(self.ball)
        hint=QLabel('不勾选第二项时，训练球心标签仅包含你手动修正过的帧，其余自动预测记为未审核，不当作真值。')
        hint.setWordWrap(True);layout.addWidget(hint)
        buttons=QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText('审核通过并保存');buttons.button(QDialogButtonBox.Ok).setEnabled(False)
        buttons.button(QDialogButtonBox.Cancel).setText('返回复核')
        self.events.toggled.connect(buttons.button(QDialogButtonBox.Ok).setEnabled)
        buttons.accepted.connect(self.accept);buttons.rejected.connect(self.reject);layout.addWidget(buttons)
