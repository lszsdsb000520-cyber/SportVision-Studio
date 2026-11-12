"""Package tested binaries, source, and a reproducible validation snapshot."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import zipfile


def archive(source,dest,prefix=None):
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for f in sorted(source.rglob('*')):
            if f.is_file():z.write(f,str(Path(prefix or source.name)/f.relative_to(source)))


def main():
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    smoke=json.loads((root/'reports/frozen-final/result.json').read_text(encoding='utf-8'))
    if not smoke.get('success'):raise RuntimeError('Frozen application has not passed validation.')
    portable=output/'PingPongStudio-Windows.zip'
    archive(root/'dist/PingPongStudio',portable)
    print('portable',portable.stat().st_size,flush=True)
    with zipfile.ZipFile(output/'PingPongStudio-source.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for folder in ('pingpong','tools','tests','sample'):
            for f in (root/folder).rglob('*'):
                if f.is_file() and '__pycache__' not in f.parts:z.write(f,str(Path('PingPongStudio-source')/f.relative_to(root)))
        for filename in ('app.py','start.cmd','README.md','SOURCES.md','pytest.ini','requirements-runtime.txt','requirements-training.txt',
                         'models/blurball.yaml','models/blurball_pretrained.pt','models/detector.onnx','models/detector.json','models/model_info.json'):
            f=root/filename;z.write(f,str(Path('PingPongStudio-source')/filename))
    validation=output/'validation';validation.mkdir(exist_ok=True)
    for name in ('data_audit/audit.json','training_result.json','frozen-final/result.json','build.json','holdout_evaluation.json'):
        source=root/'reports'/name;dest=validation/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    run=Path(json.loads((root/'reports/training_result.json').read_text(encoding='utf-8'))['run'])
    for name in ('config.json','metrics.json','result.json','curve.png'):
        shutil.copy2(run/name,validation/('training_'+name))
    shutil.copy2(root/'reports/frozen-final/app.png',output/'interface.png')
    shutil.copy2(root/'reports/frozen-final/demo_tracked.mp4',output/'demo_tracked.mp4')
    shutil.copy2(root/'reports/frozen-final/tracks.csv',output/'demo_tracks.csv')
    shutil.copy2(root/'reports/frozen-final/events.csv',output/'demo_events.csv')
    report={}
    for file in output.glob('*.zip'):
        report[file.name]={'bytes':file.stat().st_size,'sha256':hashlib.sha256(file.read_bytes()).hexdigest()}
    (output/'checksums.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
