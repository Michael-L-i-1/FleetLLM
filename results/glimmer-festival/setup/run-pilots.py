import json,subprocess,time
from pathlib import Path
r=Path("results/glimmer-festival/setup")
for j in json.loads((r/"edit-pilot-plan.json").read_text())["jobs"]:
 (r/"pilot-progress.json").write_text(json.dumps({"active":j["label"]}))
 with (r/(j["label"]+"-driver.log")).open("w") as f:rc=subprocess.run(j["command"],stdout=f,stderr=subprocess.STDOUT).returncode
 print(j["label"],rc,flush=True)
(r/"pilot-progress.json").write_text(json.dumps({"status":"complete"}))
