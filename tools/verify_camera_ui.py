"""Exercise the Start and Stop controls with a stalled camera and slow inference."""
from functools import partial
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


class SlowDetector:
    def __init__(self, *args, **kwargs):
        pass

    def run_tensor(self, tensor):
        time.sleep(.22)
        return None

    def decode_heatmaps(self, result, threshold):
        return [(None,None,0.)]*3


def main():
    from PySide6.QtWidgets import QApplication
    from pingpong.camera import CameraReader
    from pingpong.selftest import simulated_camera
    from pingpong.ui import MainWindow

    app = QApplication([])
    window = MainWindow()
    window.show()
    window.switch_mode(1)
    panel = window.live_panel
    panel.mode.setCurrentIndex(panel.mode.findData('camera'))
    panel.manual_corners = [[32,20],[124,20],[142,70],[18,70]]
    make_reader = partial(CameraReader,capture_target=simulated_camera,stall_timeout=.45)
    previews, tracked, generations = set(), set(), set()

    def pump_for(seconds):
        deadline = time.perf_counter()+seconds
        while time.perf_counter()<deadline:
            app.processEvents()
            time.sleep(.01)
            if panel.preview_key is not None:
                previews.add(panel.preview_key)
            if panel.worker and panel.worker.camera:
                generations.add(panel.worker.camera.generation)
            snapshot = panel.last_snapshot or {}
            if snapshot.get('frames'):
                tracked.add(snapshot['frames'])

    with patch('pingpong.live.CameraReader',make_reader), patch('pingpong.live.Detector',SlowDetector):
        panel.start.click()
        pump_for(8.)
        assert panel.worker and panel.worker.isRunning(),panel.status.text()
        assert len(previews)>=12 and len(tracked)>=8 and max(generations)>=2,(previews,tracked,generations)
        assert any(key[0]>=2 for key in previews),previews
        panel.stop.click()
        pump_for(1.5)
        assert panel.worker is None and panel.start.isEnabled()
        panel.start.click()
        pump_for(3.)
        assert panel.worker and panel.last_frame is not None
        panel.stop.click()
        pump_for(1.5)
        assert panel.worker is None
    window.close()
    print(json.dumps(dict(success=True,preview_updates=len(previews),tracking_updates=len(tracked),
                          driver_reconnected=max(generations)>=2,stop_and_restart=True)))


if __name__=='__main__':
    main()
