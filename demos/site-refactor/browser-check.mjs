import { chromium } from '@playwright/test';
import { resolve } from 'node:path';
import { mkdir } from 'node:fs/promises';
const [site, output] = process.argv.slice(2);
if (!site || !output) throw Error('usage: node browser-check.mjs SITE OUTPUT');
await mkdir(output, {recursive:true});
const browser = await chromium.launch({headless:true});
const errors = [];
try {
  for (const width of [390, 1440]) {
    const page = await browser.newPage({viewport:{width,height:1000}});
    await page.route('**/*', route => {
      if (new URL(route.request().url()).protocol === 'file:') return route.continue();
      errors.push(`External request: ${route.request().url()}`);
      return route.abort();
    });
    page.on('pageerror', error => errors.push(String(error)));
    await page.goto(`file://${resolve(site, 'index.html')}`);
    await page.locator('h1').waitFor();
    if (await page.locator('[data-workshop]').count() !== 3) throw Error('Expected three workshops');
    await page.locator('#search').fill('ceramics');
    if (await page.locator('[data-workshop]:visible').count() !== 1) throw Error('Search filter failed');
    await page.locator('#search').fill('no-matching-workshop');
    if (await page.locator('[data-workshop]:visible').count() !== 0) throw Error('Empty search failed');
    await page.locator('#search').fill('');
    await page.locator('#email').fill('neighbor@example.com');
    await page.locator('#signup-form button').click();
    if (!(await page.locator('#signup-status').innerText()).includes("You're on the list")) throw Error('Signup failed');
    if (await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1)) throw Error(`Horizontal overflow at ${width}px`);
    for (const id of ['search','email']) {
      if (!(await page.locator(`label[for="${id}"]`).count())) throw Error(`Missing label for ${id}`);
    }
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.screenshot({path:resolve(output, `${width}.png`), fullPage:true});
    await page.close();
  }
  if (errors.length) throw Error(errors.join('\n'));
  console.log(JSON.stringify({passed:true,widths:[390,1440]}));
} finally { await browser.close(); }
