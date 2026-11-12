from pathlib import Path
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import sys,json
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from PySide6.QtWidgets import QApplication
from pingpong.ui import MainWindow

app=QApplication([]);window=MainWindow();errors=[];window.error=errors.append;window.show()
with patch('pingpong.ui.QFileDialog.getOpenFileName',return_value=(str(ROOT/'reports/v7.1/frozen_video/project.json'),'')):
    window.open_project()
for _ in range(5):app.processEvents()
table=window.table;count=window.map_count
assert table.geometry().bottom()<count.geometry().top(),(table.geometry(),count.geometry())
assert all(rect[1]>=0 and rect[1]+rect[3]<=table.height() for rect in table.label_rects.values())
assert len(table.label_rects)==len(table.visible_events())
assert not errors,errors
window.grab().save(str(ROOT/'reports/v7.1/panel_fixed.png'))
print(json.dumps(dict(success=True,all_labels=len(table.label_rects),table_bottom=table.geometry().bottom(),controls_top=count.geometry().top(),scroll_maximum=window.right_scroll.verticalScrollBar().maximum())))
window.dirty=False;window.close()
