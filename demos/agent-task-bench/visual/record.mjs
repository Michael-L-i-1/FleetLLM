#!/usr/bin/env node
// Real development snapshots. No model calls and no generated replacement UI.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createRequire } from 'node:module';
import { spawnSync } from 'node:child_process';
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
fs.mkdirSync(output, {recursive: true});
if (fs.existsSync(path.join(output, 'frames.jsonl'))) throw Error('Refusing existing capture manifest');
fs.mkdirSync(frames, {recursive: true});
fs.writeFileSync(path.join(output, 'capture.json'), JSON.stringify({workspace, renderer, start_time_unix: start, interval_seconds: interval,
  min_change_seconds: minChangeSeconds, width: 1280, height: 800,
  note: 'Browser snapshots of actual on-disk state. Periodic/event sampling is not continuous video; no aesthetic or completion substitutions.'}, null, 2) + '\n');
const elapsed = () => Math.max(0, Date.now() / 1000 - start);
const append = x => fs.appendFileSync(path.join(output, 'frames.jsonl'), JSON.stringify(x) + '\n');
let stop = false, dirty = true, last = -Infinity, index = 0;
for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => { stop = true; });
let watcher;
try { watcher = fs.watch(workspace, {recursive: true}, () => { dirty = true; }); } catch (_) {}
function lastEvent() {
  if (!options.events) return null;
  try {
    const lines = fs.readFileSync(options.events, 'utf8').trim().split('\n');
    for (let i = lines.length - 1; i >= Math.max(0, lines.length - 10); i--) {
      try { const e = JSON.parse(lines[i]); return {event: e.event, agent: e.agent, elapsed: e.elapsed, status: e.status}; } catch (_) {}
    }
  } catch (_) {}
  return null;
}
const browser = await chromium.launch({headless: true, args: ['--disable-gpu']});
const context = await browser.newContext({viewport: {width: 1280, height: 800}, deviceScaleFactor: 1});
await context.route('**/*', async route => {
  const url = new URL(route.request().url());
  if (url.protocol === 'data:') return route.continue();
  if (url.protocol !== 'file:') return route.abort();
  try {
    const target = fs.realpathSync(fileURLToPath(url));
    const root = fs.realpathSync(preview);
    return (target === root || target.startsWith(root + path.sep)) ? route.continue() : route.abort();
  } catch (_) { return route.abort(); }
});
const page = await context.newPage();
page.on('dialog', dialog => dialog.dismiss());
async function capture(reason) {
  const t = elapsed(), rendered = spawnSync(options.python || 'python3', [renderer, '--workspace', workspace, '--output', preview],
    {encoding: 'utf8', timeout: 10000, maxBuffer: 1e6});
  if (rendered.error || rendered.status !== 0) {
    append({elapsed: t, reason, captured: false, error: String(rendered.error || rendered.stderr || rendered.stdout).slice(-1000), last_event: lastEvent()});
    return;
  }
  try {
    await page.goto(pathToFileURL(path.join(preview, 'index.html')).href, {waitUntil: 'load', timeout: 10000});
    await Promise.race([page.evaluate(() => window.scrollTo(0, 0)), new Promise((_, reject) => { const timer = setTimeout(() => reject(Error('Scroll evaluation timeout')), 5000); timer.unref(); })]);
    const filename = `frame-${String(index++).padStart(6, '0')}.png`;
    const target = path.join(frames, filename);
    await page.screenshot({path: target, animations: 'disabled', timeout: 10000});
    const item = {elapsed: elapsed(), capture_start_elapsed: t, captured: true, reason, file: path.join('frames', filename), last_event: lastEvent()};
    if (reason === 'final') {
      try {
        await page.screenshot({path: path.join(output, 'final-full-page.png'), fullPage: true, animations: 'disabled', timeout: 10000});
        item.full_page = 'final-full-page.png';
      } catch (error) { item.full_page_error = String(error).slice(-1000); }
    }
    append(item);
  } catch (error) { append({elapsed: t, reason, captured: false, error: String(error).slice(-1000), last_event: lastEvent()}); }
}
try {
  await capture('initial'); last = elapsed(); dirty = false;
  fs.writeFileSync(path.join(output, 'capture-ready.json'), JSON.stringify({elapsed: elapsed(), frames: index}) + '\n');
  while (!stop && !fs.existsSync(stopFile)) {
    await new Promise(resolve => setTimeout(resolve, 500));
    const t = elapsed();
    if (t - last >= interval || (dirty && t - last >= minChangeSeconds)) {
      await capture(dirty ? 'workspace-change' : 'periodic'); last = elapsed(); dirty = false;
    }
  }
  await capture('final');
} finally {
  watcher?.close();
  await browser.close();
  fs.writeFileSync(path.join(output, 'capture-finished.json'), JSON.stringify({elapsed: elapsed(), frames: index, stopped_by_file: fs.existsSync(stopFile)}, null, 2) + '\n');
}
