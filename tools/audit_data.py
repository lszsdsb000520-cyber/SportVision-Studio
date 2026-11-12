"""Read-only data audit and label overlay generation."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
from collections import Counter
import cv2
import numpy as np


def audit(root):
    out = root / 'reports' / 'data_audit'
    out.mkdir(parents=True, exist_ok=True)
    rows, errors, hashes = [], [], {}
    previews = []
    metadata={}
    catalog=root/'video/catalog.json'
    if catalog.exists():metadata={r['id']:r for r in json.loads(catalog.read_text(encoding='utf-8'))['videos']}
    preview_counts=Counter()
    for idx, video in enumerate(sorted((root / 'video' / 'labeled').glob('*.mp4'))):
        label = video.with_suffix('.csv')
        with label.open(encoding='utf-8-sig', newline='') as f:
            labels = list(csv.DictReader(f))
        cap = cv2.VideoCapture(str(video))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        count, visible = 0, 0
        selected = next((int(x['Frame']) for x in labels[len(labels)//3:]
                         if x['Visibility'] == '1'), -1)
        original=metadata.get(video.stem,{})
        split=original.get('split',video.stem.split('_')[0]);camera=original.get('camera',video.stem.split('_')[-1])
        preview_needed=preview_counts[(split,camera)]<2
        digest = hashlib.sha256()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if count < len(labels):
                row = labels[count]
                if int(row['Frame']) != count:
                    errors.append(f'{video.name}: frame index mismatch at {count}')
                if row['Visibility'] == '1':
                    visible += 1
                    x, y = float(row['X']), float(row['Y'])
                    if not (0 <= x < 512 and 0 <= y < 288):
                        errors.append(f'{video.name}: coordinate out of bounds at {count}')
                    if preview_needed and count == selected:
                        point = (round(x * width / 512), round(y * height / 288))
                        cv2.circle(frame, point, 12, (0, 255, 255), 2)
                        cv2.drawMarker(frame, point, (255, 0, 255), cv2.MARKER_CROSS, 24, 2)
                        cv2.putText(frame, f'{video.stem} / frame {count}', (25, 45),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        thumb = cv2.resize(frame, (768, 432))
                        cv2.imwrite(str(out / f'{video.stem}.jpg'), thumb)
                        # Separate zoom makes a tiny, fast-moving ball inspectable.
                        px, py = point
                        crop = frame[max(0, py-50):py+51, max(0, px-50):px+51]
                        cv2.imwrite(str(out / f'{video.stem}_zoom.jpg'),
                                    cv2.resize(crop, (404, 404)))
                        previews.append((video.stem, thumb))
                        preview_counts[(split,camera)]+=1
            count += 1
        cap.release()
        if count != len(labels) or count != expected:
            errors.append(f'{video.name}: decoded={count}, csv={len(labels)}, metadata={expected}')
        digest.update(video.read_bytes())
        checksum = digest.hexdigest()
        hashes.setdefault(checksum, []).append(video.name)
        rows.append(dict(file=video.name, split=split, width=width,
                         height=height, fps=fps, decoded_frames=count, label_frames=len(labels),
                         visible_frames=visible, sha256=checksum))
        if idx % 30 == 0:
            print(f'audit {idx+1}/358', flush=True)
    if previews:
        panels = [p for _, p in previews]
        while len(panels) % 2:
            panels.append(np.zeros_like(panels[0]))
        cv2.imwrite(str(out / 'contact_sheet.jpg'),
                    np.vstack([np.hstack(panels[i:i+2]) for i in range(0, len(panels), 2)]))
    duplicates = [v for v in hashes.values() if len(v) > 1]
    report = dict(videos=len(rows), decoded_frames=sum(x['decoded_frames'] for x in rows),
                  visible_frames=sum(x['visible_frames'] for x in rows),
                  splits=dict(Counter(x['split'] for x in rows)),
                  exact_duplicate_videos=duplicates, errors=errors,
                  coordinate_space=[512, 288],
                  coordinate_space_status='README claim; inspect overlay images before training',
                  caveats=['No bounce or table-corner ground truth is present.',
                           'Original match identity cannot be established from renamed files alone.',
                           'Byte duplicate check does not exclude overlapping or synchronized clips.'],
                  files=rows)
    (out / 'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='files'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    audit(parser.parse_args().root)
