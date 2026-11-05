// Capture initial-page previews independently of functional checks (including failed runs).
import { chromium } from '@playwright/test';
import { resolve } from 'node:path';
import { mkdir } from 'node:fs/promises';
const [site, output] = process.argv.slice(2);
if (!site || !output) throw Error('usage: node capture.mjs SITE OUTPUT');
await mkdir(output, {recursive:true});
const browser = await chromium.launch({headless:true});
try {
  for (const width of [390, 1440]) {
    const page = await browser.newPage({viewport:{width,height:1000}});
    await page.route('**/*', route => new URL(route.request().url()).protocol === 'file:' ? route.continue() : route.abort());
    await page.goto(`file://${resolve(site,'index.html')}`);
    await page.evaluate(() => window.scrollTo(0,0));
    await page.screenshot({path:resolve(output,`${width}.png`),fullPage:true});
    await page.close();
  }
} finally { await browser.close(); }
