from pathlib import Path
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import sys,time,json,cProfile,pstats,io
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from PySide6.QtWidgets import QApplication
from pingpong.ui import MainWindow

def run(output,seconds=6):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    app=QApplication([]);w=MainWindow();errors=[];w.error=errors.append;w.show()
    with patch('pingpong.ui.QFileDialog.getOpenFileName',return_value=(str(ROOT/'reports/v7.1/final_video/project.json'),'')):
        w.open_project()
    assert not errors,errors
    profiler=cProfile.Profile();profiler.enable()
    start=time.perf_counter();w.toggle_play()
    while time.perf_counter()-start<seconds:
        app.processEvents();time.sleep(.001)
    elapsed=time.perf_counter()-start;w.pause();profiler.disable()
    playback_time=w.project['tracks'][w.frame_index]['time']
    result=dict(wall_seconds=elapsed,video_seconds=playback_time,position_frame=w.frame_index,video_fps=w.fps,
                realtime_ratio=playback_time/elapsed,errors=errors)
    stream=io.StringIO();pstats.Stats(profiler,stream=stream).sort_stats('cumulative').print_stats(28)
    (output/'profile.txt').write_text(stream.getvalue(),encoding='utf-8')
    (output/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
    w.dirty=False;w.close()

if __name__=='__main__':run(sys.argv[1])
