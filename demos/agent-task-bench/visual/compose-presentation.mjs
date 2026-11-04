#!/usr/bin/env node
// Minimal presentation restyle of a three-lane real-time comparison.
//
// Honesty rules enforced here (same as compose.mjs):
//  * Every lane shares one real-time axis and one acceleration factor.
//  * Each lane shows the genuine per-page screenshot (config.page, default the home
//    page), or follows the original capture-selected page, from the latest manifest frame whose elapsed <= t; once t reaches
//    the lane's real duration it freezes on the final captured frame.
//  * Throughput numbers are read from summary.json at render time (workflow average
//    output tokens/s over the whole run), never animated or interpolated.
//  * No source image is modified, recaptured or synthesised; every image used is
//    listed with its path and SHA-256 in provenance.json.
//
// usage: node compose-presentation.mjs CONFIG.json OUTPUT_DIR [--stills-only]
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';
import { spawnSync } from 'node:child_process';

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, '../../..');
const require = createRequire(path.resolve(here, '../../site-refactor/package.json'));
const { chromium } = require('@playwright/test');

const args = process.argv.slice(2);
const stillsOnly = args.includes('--stills-only');
const [configArg, outputArg] = args.filter(a => !a.startsWith('--'));
if (!configArg || !outputArg) throw Error('usage: node compose-presentation.mjs CONFIG.json OUTPUT_DIR [--stills-only]');

const configFile = path.resolve(configArg), configDir = path.dirname(configFile);
const config = JSON.parse(fs.readFileSync(configFile, 'utf8'));
const output = path.resolve(outputArg);
const rel = p => path.relative(repo, p);
const sha256 = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const resolveCfg = p => path.resolve(configDir, p);

const speed = Number(config.speed ?? 40), fps = Number(config.fps ?? 30), hold = Number(config.end_hold_seconds ?? 3);
const W = Number(config.width ?? 1920), H = Number(config.height ?? 1080);
const pageName = config.page || 'index.html';
const followCapture = config.page_mode === 'follow-capture';
if (![speed, fps, hold, W, H].every(x => Number.isFinite(x) && x > 0)) throw Error('Invalid timeline settings');
if (!Array.isArray(config.lanes) || config.lanes.length !== 3) throw Error('Exactly three lanes are required');

// ---------- inputs ----------
const summaryFile = resolveCfg(config.summary);
const summary = JSON.parse(fs.readFileSync(summaryFile, 'utf8'));
const inputs = [{role: 'config', path: rel(configFile), sha256: sha256(configFile)},
                {role: 'summary', path: rel(summaryFile), sha256: sha256(summaryFile)}];

const lanes = config.lanes.map((cfg, li) => {
  const row = summary.rows.find(r => r.mode === cfg.mode);
  if (!row) throw Error(`summary.json has no row for mode ${cfg.mode}`);
  const tps = row.workflow_output_tokens_per_second;
  const duration = row.capture_seconds;
  if (!Number.isFinite(tps) || !Number.isFinite(duration) || duration <= 0) throw Error(`Bad metrics for ${cfg.mode}`);
  const tpsDisplay = tps.toFixed(1);
  if (cfg.expected_display_tps != null && cfg.expected_display_tps !== tpsDisplay)
    throw Error(`${cfg.mode}: summary gives ${tpsDisplay} tok/s, config expected ${cfg.expected_display_tps}`);

  const dir = resolveCfg(cfg.capture_dir), manifest = path.join(dir, 'frames.jsonl');
  inputs.push({role: `manifest:${cfg.mode}`, path: rel(manifest), sha256: sha256(manifest)});
  const records = fs.readFileSync(manifest, 'utf8').trim().split('\n').map((x, i) => ({...JSON.parse(x), manifest_line: i + 1}));
  const frames = [];
  let lastGood = null;
  for (const r of records.filter(r => r.captured).sort((a, b) => a.elapsed - b.elapsed)) {
    const selectedPage = followCapture ? (r.selected_page || pageName) : pageName;
    const pg = (r.pages || []).find(p => p.name === selectedPage);
    let image = null, fallback = false;
    if (pg && pg.file && !pg.error) {
      const abs = path.resolve(dir, pg.file);
      if (!fs.existsSync(abs)) throw Error(`Missing page screenshot ${abs}`);
      image = {abs, page: pg.name, file: pg.file, page_html_sha256: pg.sha256 ?? null, page_elapsed: pg.elapsed ?? null};
      lastGood = image;
    } else if (lastGood) { image = lastGood; fallback = true; }
    if (!image) continue;
    frames.push({elapsed: r.elapsed, reason: r.reason, manifest_line: r.manifest_line, frame_file: r.file ?? null, image, fallback});
  }
  if (!frames.length) throw Error(`No genuine ${pageName} screenshots for ${cfg.mode}`);
  const finalRecord = records.find(r => r.reason === 'final');
  return {...cfg, index: li, row, tps, tpsDisplay, duration, dir, frames,
          final_capture_ok: !!(finalRecord && finalRecord.captured),
          final_uses_fallback: frames[frames.length - 1].fallback};
});

const maxDuration = Math.max(...lanes.map(l => l.duration));

function laneState(l, t) {
  const ended = t >= l.duration;
  let f = l.frames[0];
  if (ended) f = l.frames[l.frames.length - 1];          // freeze on the real final capture
  else for (const x of l.frames) { if (x.elapsed <= t) f = x; else break; }
  return {ended, shown_seconds: Math.min(t, l.duration), frame: f};
}

// ---------- timeline: one composed frame per output video frame ----------
const K = Math.ceil(maxDuration / speed * fps);           // frames 0..K-1 advance; frame K = all-final
const timeline = [];
for (let k = 0; k <= K; k++) timeline.push({video_index: k, real_seconds: Math.min(k * speed / fps, maxDuration)});
timeline[K].real_seconds = maxDuration;

// QA stills: start, first frame after each lane finishes, midpoint, final.
const stillTimes = new Map([[0, 'first'], [K, 'final']]);
stillTimes.set(timeline.findIndex(x => x.real_seconds >= maxDuration / 2), 'mid');
for (const l of lanes) {
  const k = timeline.findIndex(x => x.real_seconds >= l.duration);
  if (!stillTimes.has(k)) stillTimes.set(k, `${l.mode}-finished`);
}

// ---------- page ----------
fs.mkdirSync(output, {recursive: true});
const renderDir = path.join(output, '.render');
if (fs.existsSync(renderDir)) fs.rmSync(renderDir, {recursive: true});
fs.mkdirSync(renderDir);
const qaDir = path.join(output, 'qa');
fs.mkdirSync(qaDir, {recursive: true});

const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
const clock = s => { const m = Math.floor(s / 60), sec = Math.floor(s % 60); return `${m}:${String(sec).padStart(2, '0')}`; };

const html = `<!doctype html><meta charset="utf-8"><style>
:root{--bg:#111315;--ink:#edf0ee;--mute:rgba(237,240,238,.64);--faint:rgba(237,240,238,.48);
  --line:rgba(255,255,255,.12);--accent:#b5c9bf;--accent-soft:rgba(181,201,191,.06)}
*{box-sizing:border-box;margin:0}
html,body{width:${W}px;height:${H}px;overflow:hidden}
body{background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Helvetica Neue",Arial,sans-serif;
  -webkit-font-smoothing:antialiased;font-feature-settings:"tnum" 1}
.stage{position:absolute;inset:0;padding:28px 36px 20px;display:flex;flex-direction:column}
header{display:flex;justify-content:space-between;align-items:center;height:52px;flex:none}
h1{font-size:30px;font-weight:700;letter-spacing:-.025em;line-height:1}
.model{margin-top:7px;font-size:16px;font-weight:500;color:var(--mute)}
.clock{display:flex;align-items:baseline;gap:12px}
.clock .t{font-size:22px;font-weight:600;font-variant-numeric:tabular-nums}
.pill{font-size:13px;font-weight:600;color:var(--mute)}
main{flex:1;display:grid;grid-template-columns:repeat(3,1fr);gap:24px;align-content:center;padding-bottom:12px}
.lane{display:flex;flex-direction:column}
.head{display:flex;justify-content:space-between;align-items:baseline;min-height:28px;gap:8px}
.name{font-size:20px;font-weight:700;letter-spacing:-.01em;white-space:nowrap}
.name span{font-weight:600;color:var(--mute);font-size:16px}
.name sup{font-size:12px;color:var(--mute);margin-left:3px}
.time{font-size:12px;font-weight:600;color:var(--faint);margin-top:8px;font-variant-numeric:tabular-nums}
.time.done{color:var(--mute)}
.tps{display:flex;align-items:baseline;gap:6px;white-space:nowrap}
.tps b{font-size:22px;font-weight:700;letter-spacing:-.02em;line-height:1;color:var(--ink)}
.tps i{font-style:normal;font-size:13px;font-weight:600;color:var(--mute)}
.shot{margin-top:12px;aspect-ratio:1280/800;border-radius:7px;overflow:hidden;background:#fff;border:1px solid var(--line)}
.lane.accent .shot{border-color:rgba(181,201,191,.38)}
.shot img{display:block;width:100%;height:100%;object-fit:contain;object-position:top left}
.rail{position:relative;margin-top:12px;height:2px;border-radius:2px;background:var(--line)}
.rail .fill{position:absolute;left:0;top:0;bottom:0;border-radius:2px;background:rgba(237,240,238,.46)}
.lane.accent .rail .fill{background:var(--accent)}
.rail .end{position:absolute;top:50%;width:5px;height:5px;margin:-2.5px 0 0 -2.5px;border-radius:50%;background:var(--mute);display:none}
.rail .end.on{display:block}
footer{flex:none;display:flex;justify-content:space-between;gap:32px;font-size:13px;font-weight:500;color:var(--mute)}
</style>
<div class="stage">
<header>
  <div><h1>${esc(config.title || 'FleetPi')}</h1><div class="model">${esc(config.model || '')}</div></div>
  <div class="clock"><span class="t" id="clock">0:00</span><span class="pill">${esc(speed)}×</span></div>
</header>
<main>${lanes.map((l, i) => `
  <section class="lane${l.accent ? ' accent' : ''}">
    <div class="head"><div class="name">${esc(l.name)} <span>· ${esc(l.agents)}</span>${l.mark ? `<sup>${esc(l.mark)}</sup>` : ''}</div><div class="tps"><b>${esc(l.tpsDisplay)}</b><i>avg output tok/s</i></div></div>
    <div class="shot"><img id="img${i}" alt=""></div>
    <div class="rail"><div class="fill" id="fill${i}"></div><div class="end" id="end${i}" style="left:${(l.duration / maxDuration * 100).toFixed(4)}%"></div></div>
    <div class="time" id="time${i}">0:00</div>
  </section>`).join('')}
</main>
${config.footnote_mark || config.footnote_method ? `<footer><div>${esc(config.footnote_mark || '')}</div><div>${esc(config.footnote_method || '')}</div></footer>` : ''}
</div>`;
const htmlFile = path.join(output, 'presentation.html');
fs.writeFileSync(htmlFile, html);

const browser = await chromium.launch({headless: true, args: ['--disable-gpu']});
const page = await browser.newPage({viewport: {width: W, height: H}, deviceScaleFactor: 1});
await page.route('**/*', route => {
  const u = new URL(route.request().url());
  if (u.protocol === 'data:') return route.continue();
  if (u.protocol === 'file:') {
    const p = path.resolve(fileURLToPath(u));
    if (p === htmlFile || lanes.some(l => p.startsWith(l.dir + path.sep))) return route.continue();
  }
  return route.abort();
});
await page.goto(pathToFileURL(htmlFile).href);
await page.evaluate(() => document.fonts.ready.then(() => true));

const used = new Map();   // abs image path -> provenance record
const frameLog = [];
try {
  const indices = stillsOnly ? [...stillTimes.keys()].sort((a, b) => a - b) : timeline.map(x => x.video_index);
  for (const k of indices) {
    const t = timeline[k].real_seconds;
    const states = lanes.map(l => laneState(l, t));
    await page.evaluate(({t, states, clockText}) => {
      document.getElementById('clock').textContent = clockText;
      states.forEach((s, i) => {
        const img = document.getElementById('img' + i);
        if (img.dataset.src !== s.src) { img.dataset.src = s.src; img.src = s.src; }
        const time = document.getElementById('time' + i);
        time.textContent = s.clock;   // no check mark: finishing is not the same as passing every round
        time.classList.toggle('done', s.ended);
        document.getElementById('fill' + i).style.width = (s.fill * 100).toFixed(4) + '%';
        document.getElementById('end' + i).classList.toggle('on', s.ended);
      });
    }, {t, clockText: clock(t), states: states.map((s, i) => ({ended: s.ended, clock: clock(s.shown_seconds),
      fill: s.shown_seconds / maxDuration, src: pathToFileURL(s.frame.image.abs).href}))});
    await page.evaluate(() => Promise.all([...document.images].map(i => i.decode())));
    const target = path.join(renderDir, `${String(k).padStart(6, '0')}.png`);
    await page.screenshot({path: target});
    if (stillTimes.has(k)) fs.copyFileSync(target, path.join(qaDir, `${String(k).padStart(4, '0')}-${stillTimes.get(k)}.png`));
    frameLog.push({video_index: k, video_seconds: +(k / fps).toFixed(4), real_seconds: +t.toFixed(3),
      lanes: states.map((s, i) => {
        const im = s.frame.image, l = lanes[i];
        let u = used.get(im.abs);
        if (!u) used.set(im.abs, u = {lane: l.mode, page: im.page, path: rel(im.abs), sha256: sha256(im.abs),
          page_html_sha256: im.page_html_sha256, page_screenshot_elapsed: im.page_elapsed,
          first_shown_real_seconds: +t.toFixed(3), last_shown_real_seconds: +t.toFixed(3), source_frames: []});
        u.last_shown_real_seconds = +t.toFixed(3);
        if (!u.source_frames.includes(s.frame.manifest_line)) u.source_frames.push(s.frame.manifest_line);
        return {mode: l.mode, ended: s.ended, shown_real_seconds: +s.shown_seconds.toFixed(3),
          manifest_line: s.frame.manifest_line, manifest_elapsed: +s.frame.elapsed.toFixed(3), reason: s.frame.reason,
          page: im.page, image: rel(im.abs), fallback_to_last_good: s.frame.fallback};
      })});
  }
  fs.copyFileSync(path.join(renderDir, `${String(K).padStart(6, '0')}.png`), path.join(output, 'poster.png'));
} finally { await browser.close(); }

const videoSeconds = (K + 1) / fps + hold;
const suffix = stillsOnly ? '-stills' : '';   // a stills preview never overwrites full-render metadata
const common = {speed, fps, end_hold_seconds: hold, page: pageName, page_mode: followCapture ? 'follow-capture' : 'fixed', real_duration_seconds: maxDuration,
  comparison_video_seconds: +((K + 1) / fps).toFixed(4), total_video_seconds: +videoSeconds.toFixed(4)};
fs.writeFileSync(path.join(output, `timeline${suffix}.json`), JSON.stringify({...common,
  rule: 'Video frame k shows real time min(k*speed/fps, max). Each lane shows its latest manifest frame with elapsed <= t; at t >= lane duration it freezes on the final capture. Same speed for every lane; last frame held end_hold_seconds.',
  stills_only: stillsOnly, frames: frameLog}, null, 2) + '\n');
fs.writeFileSync(path.join(output, `provenance${suffix}.json`), JSON.stringify({...common,
  generated_by: rel(fileURLToPath(import.meta.url)), inputs,
  lanes: lanes.map(l => ({mode: l.mode, label: `${l.name} · ${l.agents}`, summary_label: l.row.label,
    workflow_output_tokens_per_second: l.tps, displayed_tps: l.tpsDisplay,
    duration_seconds_capture_clock: l.duration, task_seconds: l.row.task_seconds,
    checks: `${l.row.checks_passed}/${l.row.checks_total}`, passed_all_rounds: l.row.passed, final_passed: l.row.final_passed,
    rounds: l.row.rounds.map(r => `${r.round}:${r.checksPassed}/${r.checksTotal}`),
    final_capture_ok: l.final_capture_ok, final_uses_fallback: l.final_uses_fallback, capture_dir: rel(l.dir)})),
  notes: [
    'TPS is the workflow-average output tokens/s over each full run (summary.json workflow_output_tokens_per_second), shown statically; it is not live decode speed.',
    'Lane clocks use the capture clock (summary capture_seconds), identical to the original comparison video; task_seconds excludes the initial screenshot hook.',
    'Only real per-page screenshots from the capture manifests are shown; no page is recaptured, restyled or synthesised. Sampling skips intermediate states.',
    'vllm1 missed rounds 1-2 (passed=false) and only its final cumulative check passed; the full report retains this result. No speed-to-success claim is made against it.',
    'In follow-capture mode, each lane displays the page selected in its genuine capture manifest, restoring original recorded page navigation.'
  ],
  images: [...used.values()]}, null, 2) + '\n');

if (stillsOnly) {
  fs.rmSync(renderDir, {recursive: true});
  console.log(JSON.stringify({stills: qaDir, poster: path.join(output, 'poster.png')}));
  process.exit(0);
}
const video = spawnSync(config.ffmpeg || 'ffmpeg', ['-hide_banner', '-loglevel', 'error', '-y',
  '-framerate', String(fps), '-i', path.join(renderDir, '%06d.png'),
  '-vf', `tpad=stop_mode=clone:stop_duration=${hold},format=yuv420p`,
  '-r', String(fps), '-c:v', 'libx264', '-preset', 'slow', '-crf', '17', '-tune', 'stillimage',
  '-movflags', '+faststart', path.join(output, 'comparison.mp4')],
  {encoding: 'utf8', timeout: 900000, maxBuffer: 1e6});
if (video.status !== 0) throw Error(video.stderr || String(video.error));
fs.rmSync(renderDir, {recursive: true});
console.log(JSON.stringify({video: path.join(output, 'comparison.mp4'), poster: path.join(output, 'poster.png'),
  qa: qaDir, real_duration_seconds: maxDuration, speed, total_video_seconds: +videoSeconds.toFixed(3)}));
