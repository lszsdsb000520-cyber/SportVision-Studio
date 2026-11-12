"""Check model presets and full analysis with the adapted model, without writing approval data."""
from pathlib import Path
import os,sys,time,json,copy
os.environ['QT_QPA_PLATFORM']='offscreen'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from PySide6.QtWidgets import QApplication
from pingpong.ui import MainWindow


def main():
    app=QApplication([]);window=MainWindow();errors=[];window.error=errors.append
    window.show();app.processEvents()
    assert window.model_preset.count()==2 and window.model_preset.currentIndex()==0 and window.model_path.name=='detector.onnx'
    assert window.threshold.value()==.5
    assert window.load_video(ROOT/'sample/demo.mp4')
    window.model_preset.setCurrentIndex(1)
    assert window.model_path.name=='detector_extended.onnx' and window.threshold.value()==.3
    window.start_analysis();started=time.monotonic()
    assert not window.model_preset.isEnabled()
    while window.task:
        app.processEvents();time.sleep(.01)
        if time.monotonic()-started>90:raise RuntimeError('Profile analysis timeout')
    assert not errors,errors
    assert window.project['model']=='detector_extended.onnx' and window.project['threshold']==.3
    tracks=copy.deepcopy(window.project['tracks']);window.model_preset.setCurrentIndex(0)
    assert window.model_path.name=='detector.onnx' and window.threshold.value()==.5
    assert window.project['tracks']==tracks and window.project['model']=='detector_extended.onnx'
    report=dict(success=True,both_models_available=True,recommended_thresholds=True,
        existing_analysis_preserved=True,adapted_full_analysis_frames=window.project['frame_count'])
    (ROOT/'reports/extended/model_profiles.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report));window.dirty=False;window.close()


if __name__=='__main__':main()
