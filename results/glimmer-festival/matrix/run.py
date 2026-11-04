import json,hashlib,subprocess,time
from pathlib import Path
r=Path("results/glimmer-festival");m=r/"matrix";plan=json.loads((m/"plan.json").read_text());done=[]
for job in plan["jobs"]:
 cmd=job["command"];cfg=Path(cmd[cmd.index("--config")+1]);task=Path(cmd[cmd.index("--task")+1])
 assert hashlib.sha256(cfg.read_bytes()).hexdigest()==job["config_sha256"]
 for name,sha in job["task_file_hashes"].items():assert hashlib.sha256((task/name).read_bytes()).hexdigest()==sha,name
 for name,sha in job["harness_hashes"].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha,name
 (m/"progress.json").write_text(json.dumps({"status":"running","active_label":job["label"],"completed":done,"total":len(plan["jobs"])},indent=2)+"\n")
 start=time.monotonic()
 with (m/(job["label"]+".log")).open("w") as f:rc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT).returncode
 out=Path(cmd[cmd.index("--output")+1]);meta=json.loads((out/"config.json").read_text()) if (out/"config.json").exists() else {};summary=json.loads((out/"task/summary.json").read_text()) if (out/"task/summary.json").exists() else {}
 item={"label":job["label"],"returncode":rc,"driver_seconds":time.monotonic()-start,**{k:summary.get(k) for k in ["passed","wall_seconds","time_to_first_passing_check_seconds","completion_tokens","completion_tokens_per_wall_second","calls","tool_errors","length_capped_calls","all_rounds_passed"]}};done.append(item);print(json.dumps(item),flush=True)
 if not meta.get("complete") or summary.get("infrastructure_error"):
  (m/"progress.json").write_text(json.dumps({"status":"stopped_infrastructure","active_label":None,"completed":done,"total":len(plan["jobs"])},indent=2)+"\n");raise SystemExit(1)
(m/"progress.json").write_text(json.dumps({"status":"complete","active_label":None,"completed":done,"total":len(plan["jobs"])},indent=2)+"\n")
