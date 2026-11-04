#!/usr/bin/env node
// A shared real-time axis: every lane is accelerated equally and freezes at its actual end.
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';
import { spawnSync } from 'node:child_process';
const here = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(path.resolve(here, '../../site-refactor/package.json'));
const { chromium } = require('@playwright/test');
const [configFile, outputArg] = process.argv.slice(2);
if (!configFile || !outputArg) throw Error('usage: node compose.mjs CONFIG.json OUTPUT_DIR');
const config = JSON.parse(fs.readFileSync(configFile, 'utf8')), output = path.resolve(outputArg);
if (!Array.isArray(config.lanes) || config.lanes.length !== 3) throw Error('Exactly three lanes are required');
const speed = Number(config.speed || 20), step = Number(config.sample_seconds || 10), fps = Number(config.fps || 12);
if (![speed, step, fps].every(x => Number.isFinite(x) && x > 0)) throw Error('Invalid timeline settings');
fs.mkdirSync(output, {recursive: true});
const compositeDir = path.join(output, 'composite-frames');
fs.mkdirSync(compositeDir, {recursive: false});
const lanes = config.lanes.map(l => {
  if (!Number.isFinite(l.duration_seconds) || l.duration_seconds < 0 || !l.status) throw Error('Every lane requires actual duration_seconds and status');
  const dir = path.resolve(l.capture_dir), records = fs.readFileSync(path.join(dir, 'frames.jsonl'), 'utf8').trim().split('\n').map(x => JSON.parse(x));
  const frames = records.filter(x => x.captured && x.file).sort((a,b) => a.elapsed - b.elapsed);
  if (!frames.length) throw Error(`No genuine screenshots: ${l.label}`);
  for (const f of frames) { f.absolute = path.resolve(dir, f.file); if (!fs.existsSync(f.absolute)) throw Error('Missing frame ' + f.absolute); }
  return {...l, dir, frames, final_capture_failed: records.some(x => x.reason === 'final' && !x.captured)};
});
const maxDuration = Math.max(...lanes.map(l => l.duration_seconds));
const times = new Set([0, maxDuration]);
for (let t = step; t < maxDuration; t += step) times.add(t);
for (const l of lanes) times.add(l.duration_seconds);
const timeline = [...times].sort((a,b) => a-b);
const browser = await chromium.launch({headless: true, args: ['--disable-gpu']});
const page = await browser.newPage({viewport: {width: 1920, height: 800}, deviceScaleFactor: 1});
await page.route('**/*', route => {
  const u = new URL(route.request().url());
  if (u.protocol === 'data:') return route.continue();
  if (u.protocol === 'file:') {
    const p = path.resolve(fileURLToPath(u));
    if (p.startsWith(output + path.sep) || lanes.some(l => p.startsWith(l.dir + path.sep))) return route.continue();
  }
  return route.abort();
});
const dashboard = `<!doctype html><meta charset="utf-8"><style>
*{box-sizing:border-box}body{margin:0;background:#10181b;color:#edf4f3;font:18px ui-monospace,SFMono-Regular,Menlo,monospace}
header{height:91px;padding:20px 30px;border-bottom:1px solid #375054;display:flex;justify-content:space-between;align-items:center}
h1{margin:0;font:600 26px system-ui}header p{font-size:15px;color:#b0c4c3;margin:8px 0 0}
main{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;padding:14px 18px}.lane{background:#182428;border:1px solid #375054;border-radius:10px;overflow:hidden}
.top{padding:16px;height:89px}.label{font:600 23px system-ui}.status{margin-top:8px;font-size:15px;color:#bed3ce}
.viewport{height:400px;background:#fff;overflow:hidden}.viewport img{display:block;width:100%;height:100%;object-fit:contain;object-position:top;background:#fff}
.meta{height:118px;padding:14px;font-size:14px;line-height:1.65;color:#b8cdca}.meta strong{color:#e4f4ed;font-weight:500}
footer{padding:0 22px;font-size:13px;color:#9eb6b2}.done{color:#8de4ad}.failed{color:#ffbd9d}
</style><header><div><h1></h1><p>Actual website snapshots · identical task · runs executed separately</p></div><div id="clock"></div></header><main>${lanes.map((_,i) => `<section class="lane"><div class="top"><div id="label${i}" class="label"></div><div id="status${i}" class="status"></div></div><div class="viewport"><img id="image${i}"></div><div class="meta"><strong id="metrics${i}"></strong><div id="round${i}"></div><div id="sample${i}"></div></div></section>`).join('')}</main><footer></footer>`;
fs.writeFileSync(path.join(output, 'dashboard.html'), dashboard);
await page.goto(pathToFileURL(path.join(output, 'dashboard.html')).href);
const index = [];
function timeLabel(t) { const m = Math.floor(t/60), sec = Math.floor(t%60); return `${m}:${String(sec).padStart(2,'0')}`; }
try {
  for (let i = 0; i < timeline.length; i++) {
    const t = timeline[i];
    const states = lanes.map(l => {
      const ended = t >= l.duration_seconds;
      const effective = Math.min(t, l.duration_seconds);
      let frame = l.frames[0];
      // Final captures can finish just after the benchmark's end while recording the
      // exact end state. Use the final snapshot only once this lane has ended.
      if (ended) frame = l.frames[l.frames.length - 1];
      else for (const f of l.frames) { if (f.elapsed <= effective) frame = f; else break; }
      return {label:l.label, ended, status:l.status, effective, snapshot:frame.elapsed,
        image:pathToFileURL(frame.absolute).href, round_info:frame.round_info || frame.last_event || {}, selected_page:frame.selected_page, final_capture_failed: ended && l.final_capture_failed, metrics:ended ? (l.metrics || '') : 'Run metrics appear at finish'};
    });
    await page.evaluate(({states,t,speed,title}) => {
      document.querySelector('h1').textContent = title;
      document.querySelector('#clock').textContent = `Real elapsed ${t.label} · ${speed}× playback`;
      document.querySelector('footer').textContent = 'Screenshots sampled during real runs; intermediate states may be skipped. All lanes share one real-time axis. Finished lanes freeze; unavailable final captures are labeled and retain the last successful snapshot.';
      states.forEach((s,i) => {
        document.querySelector('#label'+i).textContent = s.label;
        const status = document.querySelector('#status'+i); status.textContent = (s.ended ? s.status.toUpperCase() : 'RUNNING') + ` · ${s.effectiveLabel}`;
        status.className = 'status ' + (s.ended ? (/pass|success|complete/i.test(s.status) && !/fail|incomplete/i.test(s.status) ? 'done' : 'failed') : '');
        document.querySelector('#image'+i).src = s.image;
        document.querySelector('#metrics'+i).textContent = s.metrics;
        const round = s.round_info;
        document.querySelector('#round'+i).textContent = round.round == null ? '' : `Round ${round.round} · ${round.phase || ''}` + (s.selected_page ? ` · ${s.selected_page}` : '');
        document.querySelector('#sample'+i).textContent = `Actual snapshot at ${s.snapshotLabel}` + (s.final_capture_failed ? ' · FINAL CAPTURE FAILED: last successful view' : '');
      });
    }, {states:states.map(s => ({...s,effectiveLabel:timeLabel(s.effective),snapshotLabel:timeLabel(s.snapshot)})),t:{label:timeLabel(t)},speed,title:config.title || 'Muse Glimmer · real coding task comparison'});
    await page.evaluate(() => Promise.all([...document.images].map(img => img.decode())));
    const filename = `composite-${String(i).padStart(6,'0')}.png`, target = path.join(compositeDir,filename);
    await page.screenshot({path:target}); index.push({real_seconds:t,file:path.join('composite-frames',filename),lanes:states});
  }
  fs.copyFileSync(path.join(output,index[index.length-1].file),path.join(output,'poster.png'));
} finally { await browser.close(); }
let concat = '';
for (let i = 0; i < index.length; i++) {
  concat += `file 'composite-frames/${path.basename(index[i].file)}'\n`;
  const seconds = i + 1 < index.length ? (index[i+1].real_seconds-index[i].real_seconds)/speed : 3;
  concat += `duration ${Math.max(seconds,1/fps)}\n`;
}
concat += `file 'composite-frames/${path.basename(index[index.length-1].file)}'\n`;
fs.writeFileSync(path.join(output,'timeline.ffconcat'),concat);
fs.writeFileSync(path.join(output,'timeline.json'),JSON.stringify({speed, sample_seconds:step, fps, real_duration_seconds:maxDuration,
  note:'Same acceleration for all lanes; final states freeze at actual completion/failure. Last3video-seconds hold all final artifacts. Metrics are supplied from actual run summaries.',frames:index},null,2)+'\n');
const video = spawnSync(config.ffmpeg || 'ffmpeg',['-hide_banner','-loglevel','error','-y','-f','concat','-safe','0','-i','timeline.ffconcat',
  '-vf',`fps=${fps},format=yuv420p`,'-t',String(maxDuration/speed+3),'-c:v','libx264','-preset','veryfast','-crf','20','-movflags','+faststart','comparison.mp4'],{cwd:output,encoding:'utf8',timeout:600000,maxBuffer:1e6});
if (video.status !== 0) throw Error(video.stderr || String(video.error));
console.log(JSON.stringify({video:path.join(output,'comparison.mp4'),poster:path.join(output,'poster.png'),real_duration_seconds:maxDuration,speed}));
