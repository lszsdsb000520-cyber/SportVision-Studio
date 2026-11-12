from pathlib import Path
import argparse
import hashlib
import json
import shutil
import zipfile


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    result=json.loads((root/'reports/v2/frozen/result.json').read_text(encoding='utf-8'))
    assert result['success'] and result['frame_step_verified']==3 and len(result['automatic_corners'])==4
    dist=root/'dist/PingPongStudio-v2'
    with zipfile.ZipFile(out/'PingPongStudio-v2-Windows.zip','w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for f in sorted(dist.rglob('*')):
            if f.is_file():z.write(f,str(Path(dist.name)/f.relative_to(dist)))
    print('portable archived',flush=True)
    with zipfile.ZipFile(out/'PingPongStudio-v2-source.zip','w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for directory in ('pingpong','tools','tests','assets','sample'):
            for f in sorted((root/directory).rglob('*')):
                if f.is_file() and '__pycache__' not in f.parts:z.write(f,str(Path('PingPongStudio-v2-source')/f.relative_to(root)))
        for name in ('app.py','start.cmd','README.md','SOURCES.md','pytest.ini','requirements-runtime.txt','requirements-training.txt',
                     'models/blurball.yaml','models/blurball_pretrained.pt','models/detector.onnx','models/detector.json',
                     'models/model_info.json','models/table.onnx','models/table.json'):
            z.write(root/name,str(Path('PingPongStudio-v2-source')/name))
    shutil.copy2(root/'reports/v2/frozen/app.png',out/'interface.png')
    validation=out/'validation'
    shutil.copytree(root/'reports/v2',validation,dirs_exist_ok=True)
    checks={}
    for f in out.glob('*.zip'):
        with zipfile.ZipFile(f) as z:assert z.testzip() is None
        checks[f.name]=dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest())
    (out/'checksums.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2),flush=True)


if __name__=='__main__':main()
