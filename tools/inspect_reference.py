"""Inspect official reference annotations without changing supplied data."""
from pathlib import Path
import base64
import json
import urllib.request
import xml.etree.ElementTree as E
import zipfile
from concurrent.futures import ThreadPoolExecutor

ROOT=Path(__file__).resolve().parents[1]
TOKEN='C3pJEPKWQAkono7'


def request(suffix,listing=False):
    url=f'https://cloud.cs.uni-tuebingen.de/public.php/dav/files/{TOKEN}/{suffix}'
    r=urllib.request.Request(url,method='PROPFIND' if listing else 'GET',headers={
        'Depth':'1','Authorization':'Basic '+base64.b64encode((TOKEN+':').encode()).decode()})
    with urllib.request.urlopen(r,timeout=60) as f:return f.read()


def main():
    work=ROOT/'work/reference';work.mkdir(parents=True,exist_ok=True)
    items=[('all_csv_midpoint_annotations.zip',False),('00/rallies_videos/',True),('01/rallies_videos/',True)]
    with ThreadPoolExecutor(3) as pool:
        results=list(pool.map(lambda item:request(*item),items))
    for (name,listing),data in zip(items,results):
        if listing:
            t=E.fromstring(data);rows=[(n.findtext('d:href',namespaces={'d':'DAV:'}),n.findtext('.//d:getcontentlength',namespaces={'d':'DAV:'})) for n in t]
            (work/(name.split('/')[0]+'_listing.json')).write_text(json.dumps(rows,indent=2),encoding='utf-8')
            print(name,rows[:8],flush=True)
        else:
            file=work/name;file.write_bytes(data)
            with zipfile.ZipFile(file) as z:
                names=z.namelist();print('annotation files',names[:12],flush=True)
                for n in names:
                    if n.endswith('.csv'):
                        print(n,z.read(n).decode()[:1500],flush=True);break


if __name__=='__main__':main()
