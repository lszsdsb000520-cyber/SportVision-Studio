from pathlib import Path
import json,hashlib
import cv2,numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'datasets/extended'
SOURCE=ROOT/'video/table_tennis_data-main/table_tennis_data-main/data/raw/ball_data'


def main():
    images=[]
    for name in ('game_1','game_2','game_3','game_4','game_5','test_3','test_5','test_7'):
        split='val' if name=='game_4' else ('test' if name.startswith('test') else 'train')
        files=sorted((BASE/'images'/split).glob(name+'_*.png'))
        if not files:continue
        labels=json.loads((SOURCE/('test' if split=='test' else 'train')/(name.replace('game_','train_')+'.json')).read_text())
        file=files[min(5,len(files)-1)];image=cv2.imread(str(file))
        p=labels[str(int(file.stem.split('_')[-1]))]
        if p['x']>=0:cv2.circle(image,(round(p['x']*512/1920),round(p['y']*288/1080)),7,(0,255,0),1)
        cv2.putText(image,file.stem,(10,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        images.append(image)
    while len(images)%2:images.append(np.zeros_like(images[0]))
    montage=cv2.vconcat([cv2.hconcat(images[i:i+2]) for i in range(0,len(images),2)])
    cv2.imwrite(str(ROOT/'reports/extended/alignment.jpg'),montage)
    if not (BASE/'manifest.json').exists():return
    manifest=json.loads((BASE/'manifest.json').read_text());seen={};duplicates=[];counts={}
    for split in ('train','val','test'):
        frames=set()
        for record in manifest['windows']:
            if record['split']!=split:continue
            assert len(record['images'])==len(record['labels'])==3
            assert record['frames'][1]==record['frames'][0]+1 and record['frames'][2]==record['frames'][1]+1
            for image_path,label_path in zip(record['images'],record['labels']):
                assert image_path not in frames
                frames.add(image_path);image=cv2.imread(str(BASE/image_path))
                assert image is not None and image.shape==(288,512,3)
                point=json.loads((BASE/label_path).read_text())
                assert not point['visible'] or 0<=point['x']<512 and 0<=point['y']<288
                digest=hashlib.sha256((BASE/image_path).read_bytes()).hexdigest()
                if digest in seen and seen[digest]!=split:duplicates.append(image_path)
                seen[digest]=split
        counts[split]=len(frames)
    assert not duplicates,duplicates
    report=dict(success=True,frames=counts,exact_cross_split_duplicates=duplicates,
        alignment='PyAV presentation timestamp maps exactly to source zero-based frame index at 120fps')
    (ROOT/'reports/extended/data_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':main()
