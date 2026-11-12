import sys
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()

import cv2

# Decoding and inference already have dedicated workers. Avoid nested OpenCV pools.
cv2.setNumThreads(1)
from pingpong.ui import main

if __name__=='__main__':
    if len(sys.argv)>2 and sys.argv[1]=='--verify-camera':
        from pathlib import Path
        import traceback
        from pingpong.selftest import run_camera_transport
        try:sys.exit(run_camera_transport(sys.argv[2]))
        except Exception:
            output=Path(sys.argv[2]);output.mkdir(parents=True,exist_ok=True)
            (output/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
            sys.exit(1)
    if len(sys.argv)>3 and sys.argv[1]=='--verify-live':
        from pathlib import Path
        import traceback
        from pingpong.selftest import run_live_video
        try:sys.exit(run_live_video(sys.argv[2],sys.argv[3]))
        except Exception:
            output=Path(sys.argv[3]);output.mkdir(parents=True,exist_ok=True)
            (output/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
            sys.exit(1)
    if len(sys.argv)>3 and sys.argv[1]=='--verify-playback':
        from pathlib import Path
        import traceback
        from pingpong.selftest import run_playback
        try:sys.exit(run_playback(sys.argv[2],sys.argv[3]))
        except Exception:
            output=Path(sys.argv[3]);output.mkdir(parents=True,exist_ok=True)
            (output/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
            sys.exit(1)
    if len(sys.argv)>3 and sys.argv[1]=='--verify-video':
        from pathlib import Path
        import traceback
        from pingpong.selftest import run_video
        try:sys.exit(run_video(sys.argv[2],sys.argv[3]))
        except Exception:
            output=Path(sys.argv[3]);output.mkdir(parents=True,exist_ok=True)
            (output/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
            sys.exit(1)
    if len(sys.argv)>2 and sys.argv[1]=='--self-test':
        from pathlib import Path
        import traceback
        from pingpong.selftest import run
        try:
            sys.exit(run(sys.argv[2]))
        except Exception:
            output=Path(sys.argv[2]);output.mkdir(parents=True,exist_ok=True)
            (output/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
            sys.exit(1)
    sys.exit(main())
