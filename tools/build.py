"""Build the portable desktop application with the isolated runtime environment."""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import hashlib
import os


def main():
    root=Path(__file__).resolve().parents[1]
    name=sys.argv[sys.argv.index('--name')+1] if '--name' in sys.argv else 'PingPongStudio-v3'
    sample=root/'sample';sample.mkdir(exist_ok=True)
    original=root/'video/labeled/train_match1_000_oblique.mp4'
    if original.exists():shutil.copy2(original,sample/'demo.mp4')
    elif not (sample/'demo.mp4').exists():raise FileNotFoundError('Missing sample/demo.mp4')
    runtime=Path(sys.executable)
    mode='--console' if '--console-debug' in sys.argv else '--windowed'
    command=[str(runtime),'-m','PyInstaller','--noconfirm','--clean',mode,'--name',name,
             '--distpath',str(root/'dist'),'--workpath',str(root/'work/pyinstaller'),
             '--specpath',str(root/'work'),'--paths',str(root),
             '--collect-binaries','onnxruntime',
             '--collect-all','av',
             '--add-data',str(sample)+';sample',
             '--add-data',str(root/'assets')+';assets',
             '--add-data',str(root/'pingpong/vendor/BLURBALL_LICENSE.md')+';licenses',
             '--exclude-module','torch','--exclude-module','matplotlib','--exclude-module','onnx',
             '--exclude-module','pytest',str(root/'app.py')]
    if '--incremental' in sys.argv:command.remove('--clean')
    # Do not let unrelated native tools on the host PATH supply incompatible Qt/ICU DLLs.
    env=os.environ.copy()
    windir=Path(env.get('WINDIR','C:/Windows'))
    env['PATH']=os.pathsep.join([str(runtime.parent),sys.base_prefix,str(windir/'System32'),str(windir)])
    subprocess.run(command,cwd=root,check=True,env=env)
    dist=root/'dist'/name;(dist/'models').mkdir(exist_ok=True)
    shutil.copy2(root/'models/detector.onnx',dist/'models/detector.onnx')
    if (root/'models/detector.json').exists():shutil.copy2(root/'models/detector.json',dist/'models/detector.json')
    for filename in ('detector_extended.onnx','detector_extended.json'):
        if (root/'models'/filename).exists():shutil.copy2(root/'models'/filename,dist/'models'/filename)
    if (root/'models/model_info.json').exists():
        shutil.copy2(root/'models/model_info.json',dist/'models/model_info.json')
    shutil.copy2(root/'models/table.onnx',dist/'models/table.onnx')
    shutil.copy2(root/'models/table.json',dist/'models/table.json')
    for filename in ('README.md','SOURCES.md'):
        if (root/filename).exists():shutil.copy2(root/filename,dist/filename)
    licenses=dist/'licenses';licenses.mkdir(exist_ok=True)
    packages=runtime.parent.parent/'Lib/site-packages'
    for pattern in ('numpy-*.dist-info','scipy-*.dist-info','opencv_python-*.dist-info',
                    'onnxruntime_directml-*.dist-info','pyside6_essentials-*.dist-info',
                    'shiboken6-*.dist-info','av-*.dist-info'):
        for info in packages.glob(pattern):
            for source in info.rglob('*'):
                if source.is_file() and any(word in str(source.relative_to(info)).lower() for word in ('license','copying','notice','copyright')):
                    target=licenses/info.name/source.relative_to(info)
                    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    shutil.copy2(root/'pingpong/vendor/BLURBALL_LICENSE.md',licenses/'BLURBALL_LICENSE.md')
    shutil.copy2(root/'pingpong/vendor/TT3D_LICENSE.txt',licenses/'TT3D_LICENSE.txt')
    shutil.copy2(root/'assets/ExtendedOpenTTGames_LICENSE.txt',licenses/'ExtendedOpenTTGames_LICENSE.txt')
    files=list(dist.rglob('*'));size=sum(f.stat().st_size for f in files if f.is_file())
    report=dict(executable=str(dist/(name+'.exe')),size_bytes=size,
                model_sha256=hashlib.sha256((dist/'models/detector.onnx').read_bytes()).hexdigest())
    report_folder=root/'reports'/name.removeprefix('PingPongStudio-');report_folder.mkdir(parents=True,exist_ok=True)
    (report_folder/'build.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
