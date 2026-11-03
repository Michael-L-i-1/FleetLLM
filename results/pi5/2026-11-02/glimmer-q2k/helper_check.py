import json
from pathlib import Path
import resource
import subprocess
import time
h=Path.home()
command=[str(h/'run-glimmer'),'What is the capital of France? Answer in one sentence.','--n','16']
start=time.monotonic()
with (h/'q23-helper.txt').open('w') as out, (h/'q23-helper.log').open('w') as err:
 p=subprocess.run(command,stdout=out,stderr=err,timeout=180)
result=dict(command=command,seconds=time.monotonic()-start,returncode=p.returncode,peak_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024)
(h/'q23-helper.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
print((h/'q23-helper.txt').read_text(),flush=True)
print((h/'q23-helper.log').read_text(),flush=True)
raise SystemExit(p.returncode)
