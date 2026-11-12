import json
from pathlib import Path
from pingpong.media import resolve_video,cleanup_legacy_aliases


def test_legacy_project_paths_resolve_after_renaming(tmp_path):
    folder=tmp_path/'video';folder.mkdir();new=folder/'V001.mp4';new.write_bytes(b'video')
    old=folder/'long_original_name.mp4'
    (folder/'name_map.json').write_text(json.dumps(dict(renames=[dict(old=str(old),new=str(new))])),encoding='utf-8')
    assert resolve_video(old,tmp_path)==new


def test_cleanup_removes_only_redundant_mapped_alias(tmp_path):
    folder=tmp_path/'video';folder.mkdir();new=folder/'V001.mp4';new.write_bytes(b'video')
    old=folder/'old.mp4';old.write_bytes(b'video')
    unrelated=tmp_path/'outside.mp4';unrelated.write_bytes(b'video')
    (folder/'name_map.json').write_text(json.dumps(dict(renames=[dict(old=str(old),new=str(new),locked_alias=True),
        dict(old=str(unrelated),new=str(new),locked_alias=True)])),encoding='utf-8')
    cleanup_legacy_aliases(tmp_path)
    assert not old.exists() and new.exists() and unrelated.exists()
