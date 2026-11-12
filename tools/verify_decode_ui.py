from pathlib import Path
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import json,sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from PySide6.QtWidgets import QApplication
from pingpong.ui import MainWindow

def main():
    app=QApplication([]);window=MainWindow();errors=[];window.error=errors.append
    project=ROOT/'reports/v7/user_video.analysis.json'
    with patch('pingpong.ui.QFileDialog.getOpenFileName',return_value=(str(project),'')):
        window.open_project()
    assert not errors,errors
    assert window.total==7201 and window.slider.maximum()==7200
    window.seek(7200)
    assert window.frame_index==7200 and abs(window.cap.last_timestamp-60.)<1e-6
    result=dict(success=True,reopened_project=True,actual_frames=window.total,slider_maximum=window.slider.maximum(),last_frame_timestamp=window.cap.last_timestamp,errors=errors)
    (ROOT/'reports/v7/decode_ui_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result));window.dirty=False;window.close()

if __name__=='__main__':main()
