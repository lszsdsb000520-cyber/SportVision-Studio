from pathlib import Path
import sys
import json


def app_root():
    return Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parents[1]


def asset_path(name):
    base = Path(getattr(sys, '_MEIPASS', app_root()))
    return base / name


def default_model():
    external = app_root()/'models'/'detector.onnx'
    return external if external.exists() else asset_path('models/detector.onnx')


def model_threshold(path):
    try:
        info=json.loads(Path(path).with_suffix('.json').read_text(encoding='utf-8'))
        if not isinstance(info,dict):return .5
        value=float(info.get('threshold',.5))
        return value if .1<=value<=.95 else .5
    except (OSError,ValueError,TypeError):return .5
