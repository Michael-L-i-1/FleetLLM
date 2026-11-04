import hashlib,json,tarfile,shutil
from pathlib import Path
R=Path('results/glimmer-festival');(R/'archives').mkdir(exist_ok=True);(R/'local-archives').mkdir(exist_ok=True);(R/'summaries').mkdir(exist_ok=True)
def sha(data):return hashlib.sha256(data).hexdigest()
manifest=[];text_files=[]
for run in sorted((R/'runs').iterdir()):
 if not run.is_dir():continue
 files=sorted(p for p in run.rglob('*') if p.is_file());items=[]
 for p in files:
  data=p.read_bytes();entry={'path':str(p.relative_to(R)),'bytes':len(data),'sha256':sha(data)};items.append(entry)
  if p.suffix.lower()!='.png':text_files.append(p)
 archive=R/'local-archives'/f'{run.name}.tar.gz'
 with tarfile.open(archive,'w:gz') as t:
  for p in files:t.add(p,arcname=str(p.relative_to(R)),recursive=False)
 with tarfile.open(archive,'r:gz') as t:
  for item in items:assert sha(t.extractfile(item['path']).read())==item['sha256']
 manifest.append({'run':run.name,'archive':str(archive.relative_to(R)),'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive.read_bytes()),'readback_verified':True,'files':items})
 for name in ['summary.json','rounds.json']:
  p=run/'task'/name
  if p.exists():shutil.copy2(p,R/'summaries'/f'{run.name}-{name}')
 shutil.copy2(run/'config.json',R/'summaries'/f'{run.name}-config.json')
archive=R/'archives/raw-text-traces.tar.gz'
with tarfile.open(archive,'w:gz') as t:
 for p in text_files:t.add(p,arcname=str(p.relative_to(R)),recursive=False)
with tarfile.open(archive,'r:gz') as t:
 for p in text_files:assert sha(t.extractfile(str(p.relative_to(R))).read())==sha(p.read_bytes())
(R/'archives/manifest.json').write_text(json.dumps({'compact_text_archive':{'path':str(archive.relative_to(R)),'sha256':sha(archive.read_bytes()),'verified_files':len(text_files)},'full_local_archives':manifest},indent=2)+'\n')
(R/'.gitignore').write_text('/runs/\n/local-archives/\n/video/composite-frames/\n')
print('Verified compact traces and',len(manifest),'full local raw archives; compact MB',round(archive.stat().st_size/1e6,2))
