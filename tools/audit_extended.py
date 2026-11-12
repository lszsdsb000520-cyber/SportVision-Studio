from pathlib import Path
import json
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'video/table_tennis_data-main/table_tennis_data-main/data/raw'


def main():
    report=[]
    for split in ('train','test'):
        for path in sorted((DATA/'game_data'/split).glob('*.json')):
            name=path.stem;ball_name=name.replace('game_','train_')
            labels=json.loads(path.read_text());balls=json.loads((DATA/'ball_data'/split/(ball_name+'.json')).read_text())
            endings={k:v for k,v in labels.items() if v.startswith(('left_','right_')) and len(v.split())==1 and not any(t in v for t in ('forehand','backhand'))}
            counts=Counter(endings.values())
            with_ball=sum(any(str(int(k)+d) in balls for d in range(-30,31)) for k in endings)
            report.append(dict(name=name,split=split,ball_frames=len(balls),first_frame=min(map(int,balls)),
                last_frame=max(map(int,balls)),bounces=sum(v=='bounce' for v in labels.values()),net_contacts=sum(v=='net' for v in labels.values()),
                endings=dict(counts),endings_near_ball_labels=with_ball,total_endings=len(endings)))
    out=ROOT/'reports/extended';out.mkdir(exist_ok=True)
    (out/'annotation_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for row in report:print(json.dumps(row),flush=True)


if __name__=='__main__':main()
