import html,json,shutil
from pathlib import Path
R=Path('results/glimmer-festival');out=R/'website';out.mkdir(exist_ok=True)
names={'fleet-eight':'Fleet · eight agents','vllm-eight':'Default vLLM · eight agents','vllm-single':'Default vLLM · one agent'}
lanes=[];cards=[]
for mode,title in names.items():
 label=mode+'-r1';task=R/'runs'/label/'task';p=task/'summary.json'
 if not p.exists():continue
 s=json.loads(p.read_text());capture=task/'capture';dest=out/label;dest.mkdir(exist_ok=True)
 site=Path(s.get('final_check',{}).get('details',{}).get('render',{}).get('index','')).parent
 if not (site/'index.html').exists():
  checks=sorted((task/'checks').glob('*/site/index.html'));site=checks[-1].parent if checks else None
 if site:shutil.copytree(site,dest/'site',dirs_exist_ok=True)
 for filename in ['final-full-page.png','final-entry-viewport.png']:
  if (capture/filename).exists():shutil.copy2(capture/filename,dest/filename)
 details=s.get('final_check',{}).get('details',{});checks=details.get('checks',[]);score=f"{sum(x.get('passed',False) for x in checks)}/{len(checks)}"
 first=s.get('time_to_first_passing_check_seconds');duration=s['wall_seconds'];polish=next((x['wall_seconds'] for x in s.get('rounds',[]) if x['round']==3),None)
 metric=f"Final pass {first:.1f}s" if s['passed'] and first is not None else 'No passing completion'
 metric+=f" · {s['completion_tokens_per_wall_second']:.1f} output tok/s"
 if polish is not None and s['passed']:metric+=f" · Polish {polish:.1f}s"
 if not s['passed']:metric='Runner history limit; no speedup claim'
 lanes.append({'label':title,'capture_dir':str(capture),'duration_seconds':duration,'status':('passed ' if s['passed'] else 'incomplete ')+score,'metrics':metric})
 pages=['index','lineup','schedule','venues','tickets','accessibility','volunteer','faq'];links=' · '.join(f'<a href="{label}/site/{page}.html" target="_blank">{html.escape(page.title())}</a>' for page in pages)
 options=''.join(f'<option value="{label}/site/{page}.html">{html.escape(page.title())}</option>' for page in pages)
 cards.append(f'<article><h2>{html.escape(title)}</h2><p>{html.escape(metric)} · Checks {score}</p><p>{links}</p><label>Preview page <select onchange="document.getElementById(\'{mode}\').src=this.value">{options}</select></label><iframe id="{mode}" title="Actual generated {html.escape(title)} site" src="{label}/site/index.html"></iframe><details><summary>Full home-page screenshot</summary><img src="{label}/final-full-page.png" alt="Actual generated home page"></details></article>')
(R/'video-config.json').write_text(json.dumps({'title':'Muse Glimmer 30B four-bit · eight pages · three rounds','speed':40,'sample_seconds':10,'fps':12,'lanes':lanes},indent=2)+'\n')
(out/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Actual generated festival sites</title><style>body{font:15px system-ui;margin:24px;background:#101d20;color:#eef4f2}h1{font-size:26px}h2{font-size:20px}main{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:20px}article{min-width:0}a{color:#b8d8ff}iframe{width:100%;height:700px;border:1px solid #637d80;margin:14px 0;background:white}img{width:100%}select{font:inherit;padding:6px}@media(max-width:1000px){main{grid-template-columns:1fr}}</style><h1>Actual model-generated festival websites</h1><p>Original generated pages are shown without manual design edits. All modes used Glimmer 30B four-bit and the same three-round task. This gallery is a separate artifact browser; each site remains unchanged. Output tok/s includes reported reasoning, tools, checks and repairs. Follow the page links for full-width views.</p><main>'+''.join(cards)+'</main></html>')
print('Prepared',len(lanes),'completed modes; compose only once all three finish.')
