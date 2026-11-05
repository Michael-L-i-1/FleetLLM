// Full-page development timelapse of the Pi website run, rebuilt from the run's own record.
// Starting from the task's plain fixture, every logged file_write is applied in order using the
// exact content the model returned (matched to the event by sha256), the site is rendered with the
// task's own render.py, and each frame shows the whole page as it was at that moment of the run.
//   node make-timelapse.mjs   (writes video/pi-website-timelapse.mp4)
//   OUT_NAME=video-5s VIDEO_SECONDS=4 HOLD_SECONDS=1 FPS=15 node make-timelapse.mjs   (short version)
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { chromium } from '/Users/michaelli/code.tmp/GithubDevelopment/constrained-inference-agent-workflows/demos/site-refactor/node_modules/playwright/index.mjs';

const RUN = path.dirname(new URL(import.meta.url).pathname);
const TASK = '/Users/michaelli/code.tmp/GithubDevelopment/constrained-inference-agent-workflows/demos/agent-task-bench/tasks/coffee-site-v2';
const OUTDIR = path.join(RUN, process.env.OUT_NAME || 'video'), WORK = path.join(OUTDIR, 'work');
const VIDEO_SECONDS = +(process.env.VIDEO_SECONDS || 75), FPS = +(process.env.FPS || 4), HOLD_SECONDS = +(process.env.HOLD_SECONDS || 4);
if (fs.existsSync(OUTDIR)) throw Error(`${OUTDIR} exists; use a fresh output folder`);
fs.mkdirSync(WORK, { recursive: true });

const events = fs.readFileSync(path.join(RUN, 'events.jsonl'), 'utf8').trim().split('\n').map(l => JSON.parse(l));
const summary = JSON.parse(fs.readFileSync(path.join(RUN, 'summary.json'), 'utf8'));
const sha = s => crypto.createHash('sha256').update(s).digest('hex');
const bySha = new Map();
for (const f of fs.readdirSync(path.join(RUN, 'responses'))) {
  const d = JSON.parse(fs.readFileSync(path.join(RUN, 'responses', f), 'utf8'));
  try { const a = JSON.parse((d.content || '').trim()); if (a.action === 'write') bySha.set(sha(a.content), a); } catch {}
}
const writes = events.filter(e => e.event === 'file_write').map(e => {
  const a = bySha.get(e.sha256); if (!a) throw Error(`no saved content for ${e.path} @ ${e.elapsed}`);
  return { t: e.elapsed, agent: e.agent, path: e.path, content: a.content };
});
const duration = summary.wall_seconds, speed = duration / VIDEO_SECONDS;
const sectionFiles = fs.readdirSync(path.join(TASK, 'fixture/sections')).sort();

// Render every state: the plain fixture, then after each write.
const browser = await chromium.launch({ args: ['--disable-gpu'] });
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
const ws = path.join(WORK, 'workspace');
fs.cpSync(path.join(TASK, 'fixture'), ws, { recursive: true });
const states = [];
async function snapshot(t, caption, written) {
  const dir = path.join(WORK, `render-${states.length}`);
  execFileSync('python3', [path.join(TASK, 'render.py'), '--workspace', ws, '--output', dir]);
  await page.goto('file://' + path.join(dir, 'index.html'));
  const png = path.join(WORK, `site-${states.length}.png`);
  await page.screenshot({ path: png, fullPage: true });
  states.push({ t, caption, written: new Map(written), img: 'data:image/png;base64,' + fs.readFileSync(png).toString('base64') });
}
const written = new Map();
await snapshot(0, 'Plain starter site · 8 agents reading the shared brief', written);
for (const w of writes) {
  fs.writeFileSync(path.join(ws, w.path), w.content);
  written.set(path.basename(w.path), { t: w.t, agent: w.agent });
  await snapshot(w.t, `${w.agent} wrote ${path.basename(w.path)}`, written);
}

const hm = s => `${Math.floor(s / 3600)}h ${String(Math.floor(s % 3600 / 60)).padStart(2, '0')}m`;
const esc = s => s.replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
const frame = (st, t, final) => `<!doctype html><html><head><style>
body{margin:0;width:1920px;height:1080px;background:#16231d;color:#eef3ef;font-family:-apple-system,system-ui,sans-serif;display:flex}
.site{width:1100px;height:1080px;display:flex;justify-content:center;align-items:flex-start;padding:20px 0;box-sizing:border-box;background:#0f1914}
.site img{max-height:1040px;max-width:1060px;box-shadow:0 8px 30px rgba(0,0,0,.5);border-radius:6px;background:#fff}
.panel{flex:1;padding:48px 48px 40px;display:flex;flex-direction:column;gap:22px}
h1{margin:0;font-size:34px;line-height:1.2}.sub{color:#a9bfb5;font-size:19px;margin-top:6px}
.clock{font-size:76px;font-weight:700;font-variant-numeric:tabular-nums;color:#f2c48d}.clock small{display:block;font-size:18px;color:#a9bfb5;font-weight:500}
.caption{font-size:24px;min-height:62px;color:#fff}
ul{list-style:none;margin:0;padding:0;font-size:21px;display:grid;gap:8px}li{display:flex;justify-content:space-between;color:#7f978c}li.done{color:#eef3ef}li.done b{color:#7fd19b}
.foot{margin-top:auto;font-size:18px;color:#a9bfb5;line-height:1.5}.final{color:#f2c48d;font-weight:600}
</style></head><body><div class="site"><img src="${st.img}"></div><div class="panel">
<div><h1>Muse Glimmer 30B on a Raspberry Pi 5</h1><div class="sub">FleetLLM CPU · Q2_K · 8 agents redesigning one site · prefix sharing + n-gram drafting + 8-bit KV</div></div>
<div class="clock">${hm(t)}<small>real time on the Pi (timelapse, ${Math.round(speed)}× speed)</small></div>
<div class="caption">${esc(st.caption)}</div>
<ul>${sectionFiles.map(f => { const w = st.written.get(f); return `<li class="${w ? 'done' : ''}"><span>${w ? '<b>✓</b> ' : '○ '}${esc(f.replace('.html', ''))}</span><span>${w ? hm(w.t) : 'not yet'}</span></li>`; }).join('')}</ul>
<div class="foot">${final ? `<div class="final">Stopped after ${hm(duration)} · ${writes.length} of 8 sections written · acceptance checks not passing</div>` : ''}
${summary.completion_tokens.toLocaleString()} tokens generated · ${summary.calls} model calls · ${summary.completion_tokens_per_wall_second.toFixed(2)} tok/s over the whole run<br>
Each frame is the actual site rendered from the files the agents had written by that time.</div></div></body></html>`;

// One frame per 1/FPS of video; the state shown is the latest write at that real time.
await page.setViewportSize({ width: 1920, height: 1080 });
const total = VIDEO_SECONDS * FPS, hold = HOLD_SECONDS * FPS;
for (let i = 0; i < total + hold; i++) {
  const t = Math.min(duration, (i / total) * duration);
  const st = [...states].reverse().find(s => s.t <= t);
  await page.setContent(frame(st, t, i >= total));
  await page.screenshot({ path: path.join(WORK, `f-${String(i).padStart(5, '0')}.png`) });
}
await browser.close();
const mp4 = path.join(OUTDIR, 'pi-website-timelapse.mp4');
execFileSync('ffmpeg', ['-loglevel', 'error', '-y', '-framerate', String(FPS), '-i', path.join(WORK, 'f-%05d.png'),
  '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-r', '30', '-crf', '20', mp4]);
fs.copyFileSync(path.join(WORK, `f-${String(total + hold - 1).padStart(5, '0')}.png`), path.join(OUTDIR, 'final-frame.png'));
fs.writeFileSync(path.join(OUTDIR, 'timelapse.json'), JSON.stringify({ duration, speed, writes: writes.map(w => ({ t: w.t, agent: w.agent, path: w.path })) }, null, 2));
console.log(mp4);
