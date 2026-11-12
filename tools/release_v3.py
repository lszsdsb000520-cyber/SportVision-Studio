from pathlib import Path
import argparse
import hashlib
import json
import shutil
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=True)
    report = root / 'reports/v3'
    result = json.loads((report / 'frozen/result.json').read_text(encoding='utf-8'))
    assert result['success'] and result['unknown_ball_frames_excluded']
    dist = root / 'dist/PingPongStudio-v3'
    with zipfile.ZipFile(out / 'PingPongStudio-v3-Windows.zip', 'w', zipfile.ZIP_DEFLATED, compresslevel=5) as archive:
        for path in sorted(dist.rglob('*')):
            if path.is_file():
                archive.write(path, str(Path(dist.name) / path.relative_to(dist)))
    print('Portable archive complete', flush=True)
    with zipfile.ZipFile(out / 'PingPongStudio-v3-source.zip', 'w', zipfile.ZIP_DEFLATED, compresslevel=5) as archive:
        for directory in ('pingpong', 'tools', 'tests', 'assets', 'sample'):
            for path in sorted((root / directory).rglob('*')):
                if path.is_file() and '__pycache__' not in path.parts:
                    archive.write(path, str(Path('PingPongStudio-v3-source') / path.relative_to(root)))
        for name in ('app.py', 'start.cmd', 'README.md', 'SOURCES.md', 'pytest.ini',
                     'requirements-runtime.txt', 'requirements-training.txt',
                     'models/blurball.yaml', 'models/blurball_pretrained.pt',
                     'models/detector.onnx', 'models/detector.json', 'models/model_info.json',
                     'models/table.onnx', 'models/table.json'):
            archive.write(root / name, str(Path('PingPongStudio-v3-source') / name))
    shutil.copy2(report / 'frozen/app.png', out / 'interface.png')
    validation = out / 'validation'
    validation.mkdir(exist_ok=True)
    for name in ('build.json', 'gui_verification.json', 'long_video_verification.json', 'rename_verification.json'):
        shutil.copy2(report / name, validation / name)
    shutil.copy2(report / 'frozen/result.json', validation / 'standalone_selftest.json')
    checks = {}
    for path in out.glob('*.zip'):
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None
        checks[path.name] = {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (out / 'checksums.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    print(json.dumps(checks, indent=2), flush=True)


if __name__ == '__main__':
    main()
