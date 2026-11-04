import {readFile,mkdir} from 'node:fs/promises';
import {dirname,resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {chromium} from './dependencies.mjs';
import {visiblePixelContribution} from './visible-pixels.mjs';
const here=dirname(fileURLToPath(import.meta.url));
const [workspace,site,output,roundText]=process.argv.slice(2), round=Number(roundText);
await mkdir(output,{recursive:true});
const content=JSON.parse(await readFile(resolve(here,'fixture/content.json'),'utf8'));
const pages={home:'index.html',lineup:'lineup.html',schedule:'schedule.html',venues:'venues.html',tickets:'tickets.html',accessibility:'accessibility.html',volunteer:'volunteer.html',faq:'faq.html'};
const primary={home:'tickets.html',lineup:'schedule.html',schedule:'tickets.html',venues:'accessibility.html',accessibility:'venues.html',faq:'tickets.html'};
const checks=[],failures=[],errors=[],visibilityProbes=[];
async function check(name,fn){try{if(await fn()===false)throw Error('requirement not met');checks.push({name,passed:true});}catch(e){failures.push(`${name}: ${String(e.message||e)}`);checks.push({name,passed:false});}}
function normalize(text){return text.toLowerCase().replace(/[‐‑‒–—]/g,'-').replace(/\s+/g,' ');}
for(const name of Object.keys(pages)){
 const source=await readFile(resolve(workspace,'pages',name+'.html'),'utf8');
 await check(`${name}: custom CSS`,()=>/<style\b[^>]*>[\s\S]*\{[\s\S]*\}[\s\S]*<\/style>/i.test(source));
 await check(`${name}: local markup only`,()=>!/<\s*(script|iframe|object|embed)\b|\bon\w+\s*=/i.test(source)&&!/(?:src|href)\s*=\s*['"](?:https?:)?\/\/|url\(\s*['"]?(?:https?:)?\/\//i.test(source));
}
const browser=await chromium.launch({headless:true,args:['--disable-gpu']});
try{
 for(const width of [390,1440])for(const [name,file]of Object.entries(pages)){
  const page=await browser.newPage({viewport:{width,height:1000}});page.setDefaultTimeout(2500);
  await page.route('**/*',route=>{const url=new URL(route.request().url());const path=decodeURIComponent(url.pathname);if(url.protocol==='file:'&&(Object.values(pages).map(f=>resolve(site,f)).includes(path)||path===resolve(site,'assets/lantern.svg')))return route.continue();errors.push(`Unexpected request:${url}`);return route.abort();});
  page.on('pageerror',e=>errors.push(String(e)));
  await page.goto(`file://${resolve(site,file)}`);
  const tag=`${name}/${width}px`,root=page.locator('#page-'+name),cta=root.locator('[data-primary-cta]');
  await check(`${tag}: root and descriptive heading`,async()=>await root.count()===1&&await page.locator('h1').count()===1&&(await root.locator('h1').innerText()).trim().length>=12);
  await check(`${tag}: headline typography`,async()=>await root.locator('h1').evaluate(el=>parseFloat(getComputedStyle(el).fontSize))>=(width<600?28:34));
  await check(`${tag}: primary action`,async()=>{if(await cta.count()!==1||!(await cta.innerText()).trim())return false;const box=await cta.boundingBox();if(!box||box.height<40)return false;return primary[name]?await cta.getAttribute('href')===primary[name]:await cta.getAttribute('type')==='submit';});
  await check(`${tag}: brand action color`,async()=>await cta.evaluate((el,expected)=>{const s=getComputedStyle(el);return s.backgroundColor===expected||s.backgroundImage.includes(expected);},round>=2?'rgb(8, 127, 122)':'rgb(112, 72, 198)'));
  for(const[target,selector]of[['headline',`#page-${name} h1`],['primary',`#page-${name} [data-primary-cta]`]])await check(`${tag}: visible ${target}`,async()=>{try{const p=await visiblePixelContribution(page,selector,output,`${name}-${width}-${target}`);visibilityProbes.push({page:name,width,target,...p});}catch(e){visibilityProbes.push({page:name,width,target,passed:false,...e.probe,error:String(e.message||e)});throw e;}});
  await check(`${tag}: all navigation destinations`,async()=>{for(const destination of Object.values(pages))if(!await page.locator(`nav a[href="${destination}"]:visible`).count())return false;return true;});
  await check(`${tag}: valid local links`,async()=>await page.locator('a[href]').evaluateAll((links,files)=>links.every(a=>{const h=a.getAttribute('href');if(h.startsWith('mailto:'))return true;if(h.startsWith('#'))return !!document.getElementById(h.slice(1));return files.includes(h.split('#')[0]);}),Object.values(pages)));
  await check(`${tag}: no horizontal overflow`,async()=>await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
  const text=normalize(await root.innerText());
  if(name==='home')await check(`${tag}: festival identity`,()=>['lantern commons arts weekend','12-13 june 2027','alder park','brookfield'].every(t=>text.includes(t)));
  if(name==='lineup')await check(`${tag}: actual artists`,async()=>await root.locator('[data-artist]').count()===3&&content.artists.every(a=>text.includes(a.name.toLowerCase())));
  if(name==='schedule'){
   await check(`${tag}: actual program`,async()=>{if(await root.locator('[data-event]').count()!==6)return false;for(const e of content.events){const card=root.locator('[data-event]').filter({hasText:e.name});if(await card.count()!==1||await card.getAttribute('data-category')!==e.category||!(await card.innerText()).includes(e.venue)||!(await card.locator('time').getAttribute('datetime')).startsWith(e.day))return false;}return true;});
   await check(`${tag}: accessible working filter`,async()=>{if(!await root.locator('label[for="schedule-filter"]').count())return false;await root.locator('#schedule-filter').selectOption('family');if(await root.locator('[data-event]:visible').count()!==2)return false;await root.locator('#schedule-filter').selectOption('music');if(await root.locator('[data-event]:visible').count()!==2)return false;await root.locator('#schedule-filter').selectOption('all');return await root.locator('[data-event]:visible').count()===6;});
   if(await root.locator('#schedule-filter').count())await root.locator('#schedule-filter').selectOption('all');
  }
  if(name==='venues')await check(`${tag}: actual venues and address`,async()=>await root.locator('[data-venue]').count()===3&&content.venues.every(v=>text.includes(v.name.toLowerCase()))&&text.includes('18 river walk')&&await root.locator('a[href="accessibility.html"]').count()>0);
  if(name==='tickets')await check(`${tag}: pass prices and no payment`,async()=>await root.locator('[data-pass]').count()===2&&text.includes('$18')&&text.includes('$30')&&text.includes('no payment'));
  if(name==='accessibility')await check(`${tag}: access guidance`,()=>['step-free routes','quiet space','accessible toilets','access@lanterncommons.example'].every(t=>text.includes(t)));
  if(name==='volunteer')await check(`${tag}: genuine shifts`,async()=>await root.locator('label[for="volunteer-shift"]').count()>0&&await root.locator('#volunteer-shift option[value="welcome"]').count()>0&&await root.locator('#volunteer-shift option[value="workshops"]').count()>0&&text.includes('welcome crew')&&text.includes('workshop helpers'));
  if(name==='tickets'||name==='volunteer'){
   const kind=name==='tickets'?'ticket':'volunteer';
   await check(`${tag}: accessible signup`,async()=>await root.locator(`label[for="${kind}-email"]`).count()>0&&await root.locator(`#${kind}-email`).getAttribute('type')==='email'&&await root.locator(`#${kind}-email`).getAttribute('required')!==null&&await root.locator(`#${kind}-status[role="status"][aria-live="polite"]`).count()>0);
   await check(`${tag}: real local signup behavior`,async()=>{await root.locator(`#${kind}-email`).fill('visitor@example.com');if(name==='volunteer'){const value=round>=2?'family':'welcome';if(!await root.locator(`#volunteer-shift option[value="${value}"]`).count())return false;await root.locator('#volunteer-shift').selectOption(value);}await cta.click();return await root.locator(`#${kind}-form`).getAttribute('data-submitted')==='true'&&(await root.locator(`#${kind}-status`).innerText()).trim().length>15;});
  }
  if(name==='faq')await check(`${tag}: useful native FAQ`,async()=>{const faqCopy=normalize(await root.textContent());if(await root.locator('details').count()!==4||await root.locator('details summary').count()!==4||!['famil','rain','food','riverpath'].every(t=>faqCopy.includes(t)))return false;const first=root.locator('details').first();await first.locator('summary').click();return await first.getAttribute('open')!==null;});
  if(round>=2){
   await check(`${tag}: family advisory`,async()=>{const note=root.locator('[data-family-notice]');if(await note.count()!==1||!await note.isVisible())return false;const t=normalize(await note.innerText());return ['under-12s free with an adult','no booking fees','10:00-16:00'].every(x=>t.includes(x));});
   await check(`${tag}: page-specific family update`,async()=>{
    const h=root.locator('[data-family-highlight]');if(!await h.count())return false;const t=normalize(await h.allInnerTexts().then(x=>x.join(' ')));
    if(name==='home')return t.includes('family story circle')&&t.includes('clay play');
    if(name==='lineup')return t.includes('mira chen')&&t.includes('all ages');
    if(name==='schedule'){const picks=root.locator('[data-event][data-category="family"][data-family-highlight]');return await picks.count()===2&&(await picks.allInnerTexts()).every(t=>normalize(t).includes('family pick'));}
    if(name==='venues')return t.includes('story grove')&&t.includes('quiet space');
    if(name==='tickets')return t.includes('under-12s')&&t.includes('adult');
    if(name==='accessibility')return t.includes('quiet space')&&t.includes('10:00');
    if(name==='volunteer')return await root.locator('#volunteer-shift option[value="family"]').count()>0&&t.includes('family welcome')&&t.includes('10:00');
    if(name==='faq')return normalize(await root.locator('details').first().innerText()).includes('under-12s free with an adult');
    return false;
   });
  }
  if(round>=3){
   await check(`${tag}: visible keyboard focus`,async()=>{
    await cta.evaluate(el=>el.blur());
    const before=await cta.evaluate(el=>{const s=getComputedStyle(el);return {outline:[s.outlineWidth,s.outlineStyle,s.outlineColor,s.outlineOffset].join('|'),shadow:s.boxShadow};});
    // Tab into the real native control, rather than relying on programmatic
    // focus to activate :focus-visible after preceding pointer interactions.
    const sentinel=await cta.evaluateHandle(el=>{const b=document.createElement('button');b.type='button';b.tabIndex=0;b.style.cssText='position:fixed;left:0;top:0;width:1px;height:1px;opacity:0';el.before(b);b.focus();return b;});
    try{await page.keyboard.press('Tab');if(!await cta.evaluate(el=>el===document.activeElement))return false;
     return await cta.evaluate((el,before)=>{const s=getComputedStyle(el),outline=[s.outlineWidth,s.outlineStyle,s.outlineColor,s.outlineOffset].join('|');return (parseFloat(s.outlineWidth)>=2&&s.outlineStyle!=='none'&&outline!==before.outline)||(s.boxShadow!=='none'&&s.boxShadow!==before.shadow);},before);
    }finally{await sentinel.evaluate(el=>el.remove());await sentinel.dispose();}
   });
   await check(`${tag}: accessibility help strip`,async()=>{const h=root.locator('[data-help-strip]');return await h.count()===1&&await h.isVisible()&&(await h.innerText()).includes('Need a hand?')&&await h.locator('a[href="accessibility.html"]').count()>0&&await h.locator('a[href="mailto:access@lanterncommons.example"]').count()>0;});
  }
  await page.evaluate(()=>{if(document.activeElement instanceof HTMLElement)document.activeElement.blur();scrollTo(0,0);});
  await page.screenshot({path:resolve(output,`${name}-${width}.png`),fullPage:true});
  await page.close();
 }
 await check('no browser errors or unexpected requests',()=>errors.length===0);
}finally{await browser.close();}
console.log(JSON.stringify({passed:failures.length===0,round,checks,failures,errors,visibilityProbes,widths:[390,1440],pages:Object.values(pages)}));
process.exitCode=failures.length?1:0;
