"""Short media names, legacy-project resolution and workspace discovery."""
from pathlib import Path
import json
from .paths import app_root,asset_path


def workspace_root():
    base=app_root()
    for candidate in (base,base.parent,base.parent.parent):
        if (candidate/'video').is_dir():return candidate
    return base


def resolve_video(path,root=None):
    path=Path(path)
    if path.exists():return path
    roots=[Path(root)] if root else [workspace_root()]
    roots.extend(path.parents)
    for candidate in roots:
        maps=(candidate/'video/name_map.json',candidate/'name_map.json')
        for file in maps:
            if not file.exists():continue
            data=json.loads(file.read_text(encoding='utf-8'))
            for item in data.get('renames',[]):
                if item['old'].replace('\\','/').lower()==str(path).replace('\\','/').lower():
                    target=Path(item['new'])
                    if target.exists():return target
    return path


def media_details(path,root=None,sha256=None):
    root=Path(root or workspace_root());file=root/'video/catalog.json'
    if not file.exists():file=asset_path('assets/media_catalog.json')
    if file.exists():
        for item in json.loads(file.read_text(encoding='utf-8'))['videos']:
            if (sha256 and sha256==item.get('sha256')) or (not sha256 and Path(path).stem.upper()==item['id'].upper()):return item
    return dict(id=Path(path).stem,original_name=Path(path).name,camera='unknown')


def cleanup_legacy_aliases(root=None):
    """Remove only verified redundant old-name copies left by Windows file locks."""
    import hashlib
    root=Path(root or workspace_root());mapping=root/'video/name_map.json'
    if not mapping.exists():return []
    data=json.loads(mapping.read_text(encoding='utf-8'));removed=[]
    for item in data.get('renames',[]):
        if not item.get('locked_alias'):continue
        old,new=Path(item['old']),Path(item['new'])
        if not old.resolve().is_relative_to((root/'video').resolve()) or not new.resolve().is_relative_to((root/'video').resolve()):continue
        if old.exists() and new.exists():
            if hashlib.sha256(old.read_bytes()).digest()!=hashlib.sha256(new.read_bytes()).digest():continue
            try:old.unlink();removed.append(str(old))
            except PermissionError:pass
    return removed
