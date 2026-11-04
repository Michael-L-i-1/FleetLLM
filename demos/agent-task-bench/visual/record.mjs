#!/usr/bin/env node
// Real development snapshots. No model calls and no generated replacement UI.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createRequire } from 'node:module';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
const here = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(path.resolve(here, '../../site-refactor/package.json'));
const { chromium } = require('@playwright/test');
const options = {};
for (let i = 2; i < process.argv.length; i++) {
  const key = process.argv[i];
  if (!key.startsWith('--') || !process.argv[i + 1]) throw Error(`Expected --option value: ${key}`);
  options[key.slice(2)] = process.argv[++i];
}
for (const key of ['workspace', 'output', 'stop-file']) if (!options[key]) throw Error(`Missing --${key}`);
const workspace = path.resolve(options.workspace), output = path.resolve(options.output);
const stopFile = path.resolve(options['stop-file']);
const interval = Number(options.interval || 10), minChangeSeconds = Number(options['min-change-seconds'] || 3);
const start = Number(options['start-time-unix'] || Date.now() / 1000);
if (![interval, minChangeSeconds, start].every(Number.isFinite) || interval < 1 || minChangeSeconds < 1) throw Error('Invalid timing arguments');
const renderer = path.resolve(options.renderer || path.resolve(here, '../tasks/coffee-site/render.py'));
const preview = path.join(output, 'preview'), frames = path.join(output, 'frames');
const pageNames = (options.pages || 'index.html').split(',').map(x => x.trim()).filter(Boolean);
if (!pageNames.length || new Set(pageNames).size !== pageNames.length || pageNames.some(x => path.isAbsolute(x) || x.split(/[\\/]/).includes('..') || !x.endsWith('.html'))) throw Error('Invalid --pages list');
const entry = options.entry || pageNames[0];
if (!pageNames.includes(entry)) throw Error('--entry must be in --pages');
const multiple = pageNames.length > 1, pageStates = new Map(), baselines = new Map();
let selectedPage = entry, lastWriteKey = '';
fs.mkdirSync(output, {recursive: true});
if (fs.existsSync(path.join(output, 'frames.jsonl'))) throw Error('Refusing existing capture manifest');
fs.mkdirSync(frames, {recursive: true});
fs.mkdirSync(preview, {recursive: true});
fs.writeFileSync(path.join(output, 'capture.json'), JSON.stringify({workspace, renderer, start_time_unix: start, interval_seconds: interval,
  min_change_seconds: minChangeSeconds, width: 1280, height: 800, pages: pageNames, entry, multi_page_dashboard: multiple,
  note: 'Browser snapshots of actual on-disk state. Periodic/event sampling is not continuous video; no aesthetic or completion substitutions.'}, null, 2) + '\n');
const elapsed = () => Math.max(0, Date.now() / 1000 - start);
const append = x => fs.appendFileSync(path.join(output, 'frames.jsonl'), JSON.stringify(x) + '\n');
let stop = false, dirty = true, last = -Infinity, index = 0;
for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => { stop = true; });
let watcher;
try { watcher = fs.watch(workspace, {recursive: true}, () => { dirty = true; }); } catch (_) {}
function eventState() {
  const state = {event: null, round: null, round_title: '', phase: '', round_passed: null, last_write: null};
  if (!options.events) return state;
  try {
    for (const line of fs.readFileSync(options.events, 'utf8').split('\n')) {
      let e; try { e = JSON.parse(line); } catch (_) { continue; }
      state.event = e.event;
      if (e.round != null) state.round = e.round;
      if (e.round_title) state.round_title = e.round_title;
      if (e.phase) state.phase = e.phase;
      if (e.event === 'round_start') { state.round_title = e.title || state.round_title; state.round_passed = null; }
      if (e.event === 'round_complete') { state.round_title = e.title || state.round_title; state.round_passed = e.passed; }
      if (e.event === 'file_write' && e.path) state.last_write = {path: e.path, elapsed: e.elapsed, agent: e.agent, sha256: e.sha256};
    }
  } catch (_) {}
  return state;
}
const lastEvent = () => eventState();
const clock = t => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, '0')}`;

const browser = await chromium.launch({headless: true, args: ['--disable-gpu']});
const context = await browser.newContext({viewport: {width: 1280, height: 800}, deviceScaleFactor: 1});
await context.route('**/*', async route => {
  const url = new URL(route.request().url());
  if (url.protocol === 'data:') return route.continue();
  if (url.protocol !== 'file:') return route.abort();
  try {
    const target = fs.realpathSync(fileURLToPath(url));
    const roots = [fs.realpathSync(preview), fs.realpathSync(frames)];
    return (roots.some(root => target === root || target.startsWith(root + path.sep)) || (fs.existsSync(path.join(output, 'page-dashboard.html')) && target === fs.realpathSync(path.join(output, 'page-dashboard.html')))) ? route.continue() : route.abort();
  } catch (_) { return route.abort(); }
});
const page = await context.newPage();
page.on('dialog', dialog => dialog.dismiss());
const board = multiple ? await context.newPage() : null;
if (board) {
  const cards = pageNames.map((_, i) => `<article id="card${i}"><img id="thumb${i}"><div class="name" id="name${i}"></div><div class="hint" id="hint${i}"></div></article>`).join('');
  fs.writeFileSync(path.join(output, 'page-dashboard.html'), `<!doctype html><meta charset="utf-8"><style>
*{box-sizing:border-box}body{margin:0;color:#e5efec;background:#101c22;font:14px system-ui}header{height:66px;padding:10px 16px;border-bottom:1px solid #385054;display:flex;justify-content:space-between;gap:18px}h1{font-size:20px;margin:0 0 4px}#stage{color:#a8c5be;font-size:13px;max-width:1050px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}#clock{font:17px ui-monospace,monospace;padding-top:9px;white-space:nowrap}
main{height:668px;padding:14px;display:grid;grid-template-columns:minmax(0,1fr) 318px;gap:14px}.selected{border:1px solid #385054;border-radius:8px;overflow:hidden;background:#17282c}.bar{height:40px;padding:10px 13px;font-weight:600;display:flex;justify-content:space-between}.picture{height:572px;display:flex;align-items:flex-start;justify-content:center;background:#fff}.picture img{display:block;width:100%;height:100%;object-fit:contain;object-position:top}.sample{height:26px;padding:5px 12px;font:11px ui-monospace,monospace;color:#abc4bc}.overview{font-size:15px;margin:0 0 10px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:9px}.grid article{min-width:0;background:#17282c;border:1px solid #385054;border-radius:5px;overflow:hidden}.grid article.active{border-color:#9ce4c4}.grid img{display:block;width:100%;height:94px;object-fit:contain;object-position:top;background:#fff}.grid .name{padding:5px 7px 1px;font-size:12px;font-weight:600}.hint{padding:1px 7px 5px;font:10px ui-monospace,monospace;color:#abc4bc}.aside-note{font:11px/1.5 ui-monospace,monospace;color:#abc4bc;padding-top:12px}
footer{height:66px;padding:9px 16px;border-top:1px solid #385054;font-size:12px;line-height:1.7;color:#abc4bc}#lastwrite{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.error{color:#ffc098}
</style><header><div><h1>Actual workspace · ${pageNames.length} pages</h1><div id="stage"></div></div><div id="clock"></div></header><main><section class="selected"><div class="bar"><span id="selected-name"></span><span id="selected-change"></span></div><div class="picture"><img id="selected-image"></div><div class="sample" id="selected-sample"></div></section><aside><h2 class="overview" id="overview"></h2><div class="grid">${cards}</div><div class="aside-note">Thumbnails are genuine page screenshots.<br>Changed HTML is not an acceptance-test result.</div></aside></main><footer><div id="lastwrite"></div><div>Real source state · CPU browser capture · sampled development timeline; no replacement site designs</div></footer>`);
  await board.goto(pathToFileURL(path.join(output, 'page-dashboard.html')).href);
}
async function showPage(name, target) {
  await page.goto(pathToFileURL(path.join(preview, name)).href, {waitUntil: 'load', timeout: 10000});
  await Promise.race([page.evaluate(() => window.scrollTo(0, 0)), new Promise((_, reject) => { const timer = setTimeout(() => reject(Error('Scroll evaluation timeout')), 5000); timer.unref(); })]);
  await page.screenshot({path: target, animations: 'disabled', timeout: 10000});
}
async function pageDashboard(reason, state, sourceMap) {
  for (let i = 0; i < pageNames.length; i++) {
    const name = pageNames[i], previous = pageStates.get(name);
    try {
      const hash = createHash('sha256').update(fs.readFileSync(path.join(preview, name))).digest('hex');
      if (!baselines.has(name)) baselines.set(name, hash);
      if (!previous || previous.sha256 !== hash || previous.error) {
        const file = path.join('frames', `page-${String(i).padStart(2, '0')}-${String(index).padStart(6, '0')}.png`);
        await showPage(name, path.join(output, file));
        pageStates.set(name, {name, sha256: hash, file, elapsed: elapsed(), changed_html: hash !== baselines.get(name)});
      }
    } catch (error) {
      if (!baselines.has(name)) baselines.set(name, null);
      pageStates.set(name, {...previous, name, error: String(error).slice(-400)});
    }
  }
  const write = state.last_write, key = write ? `${write.path}:${write.elapsed}:${write.sha256}` : '';
  const outputPage = write && (sourceMap?.[write.path] || pageNames.find(p => path.basename(p) === path.basename(write.path)));
  if (reason === 'initial' || reason === 'final') selectedPage = entry;
  else if (outputPage && pageNames.includes(outputPage) && key !== lastWriteKey) selectedPage = outputPage;
  else selectedPage = pageNames[(pageNames.indexOf(selectedPage) + 1) % pageNames.length];
  lastWriteKey = key;
  const records = pageNames.map(name => pageStates.get(name));
  await board.evaluate(({records, selected, state, time}) => {
    const chosen = records.find(x => x.name === selected);
    document.querySelector('#clock').textContent = time;
    document.querySelector('#stage').textContent = state.round == null ? 'No round event recorded yet' : `Round ${state.round}: ${state.round_title || ''} · ${state.phase || 'starting'}` + (state.round_passed == null ? '' : ` · round check ${state.round_passed ? 'passed' : 'failed'}`);
    document.querySelector('#selected-name').textContent = `Viewing ${selected}`;
    document.querySelector('#selected-change').textContent = chosen.error ? 'capture error · last good view' : chosen.changed_html ? 'changed HTML' : 'starter HTML';
    document.querySelector('#selected-sample').textContent = chosen.elapsed_label ? `Actual page screenshot at ${chosen.elapsed_label}` : 'No successful page screenshot';
    const main = document.querySelector('#selected-image'); if (chosen.url) main.src = chosen.url; else main.removeAttribute('src');
    document.querySelector('#overview').textContent = `${records.filter(x => x.changed_html).length}/${records.length} pages with changed HTML`;
    document.querySelector('#lastwrite').textContent = state.last_write ? `Last actual write: ${state.last_write.path} · ${state.last_write.agent || ''}` : 'No file_write event recorded yet';
    records.forEach((r, i) => {
      const image = document.querySelector('#thumb'+i); if (r.url) image.src = r.url; else image.removeAttribute('src');
      document.querySelector('#name'+i).textContent = r.name;
      document.querySelector('#hint'+i).textContent = r.error ? 'capture unavailable / last view' : (r.changed_html ? 'changed HTML' : 'starter HTML') + ` · ${r.elapsed_label || ''}`;
      document.querySelector('#card'+i).classList.toggle('active', r.name === selected);
    });
  }, {records: records.map(r => ({...r, url: r.file ? pathToFileURL(path.join(output, r.file)).href : null, elapsed_label: r.elapsed == null ? null : clock(r.elapsed)})), selected: selectedPage, state, time: clock(elapsed())});
  await board.evaluate(() => Promise.all([...document.images].filter(i => i.hasAttribute('src')).map(i => i.decode())));
  return {pages: records, selected_page: selectedPage};
}
async function capture(reason) {
  const t = elapsed(), rendered = spawnSync(options.python || 'python3', [renderer, '--workspace', workspace, '--output', preview],
    {encoding: 'utf8', timeout: 10000, maxBuffer: 1e6});
  if (rendered.error || rendered.status !== 0) {
    append({elapsed: t, reason, captured: false, error: String(rendered.error || rendered.stderr || rendered.stdout).slice(-1000), last_event: lastEvent()});
    return;
  }
  try {
    const state = eventState();
    let details = {}, sourceMap = {};
    try { sourceMap = JSON.parse(rendered.stdout).source_to_output || {}; } catch (_) {}
    if (multiple) details = await pageDashboard(reason, state, sourceMap);
    else await showPage(entry, path.join(frames, `single-${String(index).padStart(6, '0')}.png`));
    const filename = `frame-${String(index).padStart(6, '0')}.png`;
    const target = path.join(frames, filename);
    if (multiple) await board.screenshot({path: target, animations: 'disabled', timeout: 10000});
    else fs.renameSync(path.join(frames, `single-${String(index).padStart(6, '0')}.png`), target);
    const item = {elapsed: elapsed(), capture_start_elapsed: t, captured: true, reason, file: path.join('frames', filename), last_event: state, round_info: state, ...details};
    if (reason === 'final') {
      try {
        if (multiple) await showPage(entry, path.join(output, 'final-entry-viewport.png'));
        await page.screenshot({path: path.join(output, 'final-full-page.png'), fullPage: true, animations: 'disabled', timeout: 10000});
        item.full_page = 'final-full-page.png';
      } catch (error) { item.full_page_error = String(error).slice(-1000); }
    }
    append(item); index++;
  } catch (error) { append({elapsed: elapsed(), reason, captured: false, error: String(error).slice(-1000), last_event: lastEvent()}); }
}

try {
  dirty = false; await capture('initial'); last = elapsed();
  fs.writeFileSync(path.join(output, 'capture-ready.json'), JSON.stringify({elapsed: elapsed(), frames: index}) + '\n');
  while (!stop && !fs.existsSync(stopFile)) {
    await new Promise(resolve => setTimeout(resolve, 500));
    const t = elapsed();
    if (t - last >= interval || (dirty && t - last >= minChangeSeconds)) {
      const reason = dirty ? 'workspace-change' : 'periodic'; dirty = false;
      await capture(reason); last = elapsed();
    }
  }
  await capture('final');
} finally {
  watcher?.close();
  await browser.close();
  fs.writeFileSync(path.join(output, 'capture-finished.json'), JSON.stringify({elapsed: elapsed(), frames: index, stopped_by_file: fs.existsSync(stopFile)}, null, 2) + '\n');
}
