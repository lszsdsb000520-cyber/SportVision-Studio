from pathlib import Path
import argparse,hashlib,json,shutil,zipfile


def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda:file.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def main():
    root=Path(__file__).resolve().parents[1];parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--version',default='v4')
    args=parser.parse_args();version=args.version;out=args.output;out.mkdir(parents=True,exist_ok=True)
    result=json.loads((root/f'reports/{version}/frozen/result.json').read_text());assert result['success']
    training=json.loads((root/'reports/extended/training_result.json').read_text());run=Path(training['run'])
    dist=root/f'dist/PingPongStudio-{version}'
    expected=json.loads((root/'reports/extended/deployment_decision.json').read_text(encoding='utf-8'))
    assert sha256(dist/'models/detector.onnx')==expected['default_model_sha256']
    assert sha256(dist/'models/detector_extended.onnx')==expected['candidate_model_sha256']
    with zipfile.ZipFile(out/f'PingPongStudio-{version}-Windows.zip','w',zipfile.ZIP_DEFLATED,compresslevel=5) as archive:
        for path in sorted(dist.rglob('*')):
            if path.is_file():archive.write(path,str(Path(dist.name)/path.relative_to(dist)))
    print('Portable archive complete',flush=True)
    with zipfile.ZipFile(out/f'PingPongStudio-{version}-source.zip','w',zipfile.ZIP_DEFLATED,compresslevel=5) as archive:
        def add(path):archive.write(path,str(Path(f'PingPongStudio-{version}-source')/path.relative_to(root)))
        for directory in ('pingpong','tools','tests','assets','sample'):
            for path in sorted((root/directory).rglob('*')):
                if path.is_file() and '__pycache__' not in path.parts:add(path)
        for name in ('app.py','start.cmd','README.md','SOURCES.md','pytest.ini','requirements-runtime.txt','requirements-training.txt',
                     'models/blurball.yaml','models/blurball_pretrained.pt','models/detector.onnx','models/detector.json',
                     'models/detector_extended.onnx','models/detector_extended.json','models/model_info.json','models/table.onnx','models/table.json',
                     'datasets/extended/manifest.json'):
            add(root/name)
        for name in ('best.pt','config.json','metrics.json','result.json','selection.json','curve.png'):add(run/name)
        source=root/'video/table_tennis_data-main/table_tennis_data-main'
        for name in ('README.md','LICENSE'):add(source/name)
        for path in (source/'data/raw').rglob('*.json'):add(path)
    models=out/'models';models.mkdir(exist_ok=True)
    for name in ('detector_extended.onnx','detector_extended.json'):shutil.copy2(root/'models'/name,models/name)
    shutil.copy2(run/'best.pt',models/'best.pt');shutil.copy2(root/'assets/ExtendedOpenTTGames_LICENSE.txt',models/'LICENSE.txt')
    shutil.copy2(root/f'reports/{version}/frozen/result.json',out/'standalone_selftest.json')
    shutil.copy2(root/f'reports/{version}/frozen/app.png',out/'interface.png')
    if (root/f'reports/{version}/interactive_validation.json').exists():
        shutil.copy2(root/f'reports/{version}/interactive_validation.json',out/'interactive_validation.json')
    shutil.copy2(root/'reports/extended/comparison.jpg',out/'comparison.jpg')
    shutil.copy2(root/'reports/extended/model_profiles.json',out/'model_profiles.json')
    checks={}
    for path in out.glob('*.zip'):
        with zipfile.ZipFile(path) as archive:assert archive.testzip() is None
        checks[path.name]=dict(bytes=path.stat().st_size,sha256=sha256(path))
    (out/'checksums.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2),flush=True)


if __name__=='__main__':main()
