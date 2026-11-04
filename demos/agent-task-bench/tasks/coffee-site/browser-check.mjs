import {fileURLToPath} from 'node:url';
import {dirname, resolve} from 'node:path';
import {readFile, mkdir} from 'node:fs/promises';
const here = dirname(fileURLToPath(import.meta.url));
const {chromium} = await import(resolve(here, '../../../site-refactor/node_modules/@playwright/test/index.mjs'));
const [workspace, site, output] = process.argv.slice(2);
await mkdir(output, {recursive:true});
const checks = [], failures = [], errors = [];
async function check(name, action) {
  try { if (await action() === false) throw Error('requirement not met'); checks.push({name, passed:true}); }
  catch (error) { failures.push(`${name}: ${String(error.message || error)}`); checks.push({name, passed:false}); }
}
const names = ['01-header.html','02-hero.html','03-mission.html','04-menu.html',
  '05-impact.html','06-events.html','07-visit.html','08-footer.html'];
for (const name of names) {
  const text = await readFile(resolve(workspace, 'sections', name), 'utf8');
  await check(`${name}: custom styling supplied`, () => /<style\b[^>]*>[\s\S]*\{[\s\S]*\}[\s\S]*<\/style>/i.test(text));
  await check(`${name}: markup only`, () => !/<\s*(script|iframe|object|embed)\b|\bon\w+\s*=/i.test(text));
  await check(`${name}: no remote assets`, () => !/(?:src|href)\s*=\s*['"](?:https?:)?\/\/|url\(\s*['"]?(?:https?:)?\/\//i.test(text));
}
const browser = await chromium.launch({headless:true, args:['--disable-gpu']});
try {
  for (const width of [390,1440]) {
    const page = await browser.newPage({viewport:{width,height:1000}});
    await page.route('**/*', route => {
      const url = new URL(route.request().url());
      if (url.protocol === 'file:' && decodeURIComponent(url.pathname) === resolve(site,'index.html')) return route.continue();
      errors.push(`Unexpected request: ${url}`); return route.abort();
    });
    page.on('pageerror', error => errors.push(String(error)));
    await page.goto(`file://${resolve(site,'index.html')}`);
    const tag = `${width}px`;
    await check(`${tag}: one descriptive h1`, async () => await page.locator('h1').count() === 1 && (await page.locator('h1').innerText()).trim().length >= 12);
    await check(`${tag}: hero typography`, async () => await page.locator('#hero h1').evaluate(el => parseFloat(getComputedStyle(el).fontSize)) >= (width<600 ? 28:40));
    await check(`${tag}: navigation`, async () => {
      if (!await page.getByRole('navigation').count()) return false;
      for (const id of ['menu','mission','events','visit']) if (!await page.locator(`nav a[href="#${id}"]:visible`).count()) return false;
      return true;
    });
    await check(`${tag}: section headings`, async () => {
      for (const id of ['mission','menu','impact','events','visit','newsletter']) if (!await page.locator(`#${id} h2`).count()) return false;
      return true;
    });
    await check(`${tag}: hero calls to action`, async () => await page.locator('#hero a[href="#menu"]').count()>0 && await page.locator('#hero a[href="#mission"]').count()>0);
    await check(`${tag}: mission disclosure`, async () => { const t=(await page.locator('#mission').innerText()).toLowerCase().replace(/\s+/g,' ');return t.includes('100%') && t.includes('job training') && t.includes('nonprofit'); });
    await check(`${tag}: menu cards`, async () => {
      if (await page.locator('[data-menu-item]').count() !== 3) return false;
      for (const [text,price] of [['Espresso','$3'],['Oat Latte','$5'],['Cold Brew','$4']]) {
        const card=page.locator('[data-menu-item]').filter({hasText:text});
        if (await card.count() !== 1 || !(await card.innerText()).includes(price)) return false;
      }
      return true;
    });
    await check(`${tag}: labelled search`, async () => await page.locator('label[for="menu-search"]').count()>0 && await page.locator('#menu-search').getAttribute('type') === 'search');
    await check(`${tag}: search behavior`, async () => {
      await page.locator('#menu-search').fill('oat');
      if (await page.locator('[data-menu-item]:visible').count() !== 1) return false;
      await page.locator('#menu-search').fill('no-such-beverage');
      if (await page.locator('[data-menu-item]:visible').count() !== 0 || !await page.locator('#menu-empty').isVisible()) return false;
      await page.locator('#menu-search').fill('');
      return await page.locator('[data-menu-item]:visible').count() === 3;
    });
    // Always reset, including after a failed assertion, for fair independent checks.
    if (await page.locator('#menu-search').count()) await page.locator('#menu-search').fill('');
    await check(`${tag}: impact facts`, async () => { const t=await page.locator('#impact').innerText();return ['2,500','120','35','meals','training','graduates'].every(x=>t.toLowerCase().includes(x)); });
    await check(`${tag}: dated events`, async () => {
      if (await page.locator('[data-event]').count() !== 3) return false;
      const expected=[['Community Table','2027-03-20'],['Open Mic','2027-04-03'],['Coffee Skills','2027-04-17']];
      for (const [name,date] of expected) {
        const el=page.locator('[data-event]').filter({hasText:name});if (await el.count() !== 1) return false;
        if (!await el.locator('h3').count() || !await el.locator('a[href="#newsletter"]').count()) return false;
        const d=await el.locator('time').getAttribute('datetime');if (!d || Number.isNaN(Date.parse(d)) || d.slice(0,10)!==date) return false;
      }
      return true;
    });
    await check(`${tag}: visit details`, async () => { const t=await page.locator('#visit').innerText();return await page.locator('#visit address').count()>0 && t.includes('42 Willow') && t.includes('Palo Alto') && t.includes('07:00') && t.includes('18:00') && t.includes('08:00') && t.includes('16:00'); });
    await check(`${tag}: accessible newsletter`, async () => await page.locator('label[for="email"]').count()>0 && await page.locator('#email').getAttribute('type')==='email' && await page.locator('#email').getAttribute('required')!==null && await page.locator('#newsletter-status[role="status"][aria-live="polite"]').count()>0);
    await check(`${tag}: newsletter behavior`, async () => {
      await page.locator('#email').fill('neighbor@example.com');
      await page.locator('#newsletter-form button[type="submit"]').click();
      return (await page.locator('#newsletter-status').innerText()).includes('Thanks');
    });
    await check(`${tag}: valid internal anchors`, async () => await page.locator('a[href^="#"]').evaluateAll(els => els.every(el => el.hash.length>1 && document.getElementById(decodeURIComponent(el.hash.slice(1))))));
    await check(`${tag}: no horizontal overflow`, async () => await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth+1));
    await page.evaluate(() => scrollTo(0,0));
    await page.screenshot({path:resolve(output,`${width}.png`),fullPage:true});
    await page.close();
  }
  await check('no JavaScript errors or external requests', () => errors.length===0);
} finally { await browser.close(); }
console.log(JSON.stringify({passed:failures.length===0, checks, failures, errors, widths:[390,1440]}));
process.exitCode = failures.length ? 1:0;
