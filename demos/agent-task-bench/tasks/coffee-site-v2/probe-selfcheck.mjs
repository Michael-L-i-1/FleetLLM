// Model-free evaluator regression checks. These are not benchmark solutions.
import {fileURLToPath} from 'node:url';
import {dirname, resolve} from 'node:path';
import {mkdir, mkdtemp} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {visiblePixelContribution} from './visible-pixels.mjs';
const here=dirname(fileURLToPath(import.meta.url));
const {chromium}=await import(resolve(here,'../../../site-refactor/node_modules/@playwright/test/index.mjs'));
const out=process.argv[2] || await mkdtemp(resolve(tmpdir(),'coffee-pixel-selfcheck-'));
await mkdir(out,{recursive:true});
const browser=await chromium.launch({headless:true,args:['--disable-gpu']});
const results=[];
try {
  const page=await browser.newPage({viewport:{width:800,height:500}});
  const cases=[
    {name:'visible',html:'<div class="wrap"><h1 id="target" style="color: #123; margin: 0">Visible coffee headline</h1></div>',expected:true},
    {name:'pointer-events-none-occlusion',html:'<div class="wrap"><h1 id="target">Hidden coffee headline</h1></div>',css:'.wrap::before{content:"";position:absolute;inset:0;background:#fff;pointer-events:none}',expected:false},
    {name:'transparent-target',html:'<div class="wrap"><h1 id="target" style="opacity:0">Transparent coffee headline</h1></div>',expected:false},
    {name:'explicit-visible-child',html:'<div class="wrap"><h1 id="target" style="color: #123; margin: 0"><span style="visibility:visible!important">Nested visible coffee headline</span></h1></div>',expected:true},
    {name:'animated-opaque-occlusion',html:'<div class="wrap"><h1 id="target">Hidden by animated decoration</h1></div>',css:'@keyframes paint{from{background:#fff}to{background:#eeffee}}.wrap::before{content:"";position:absolute;inset:0;pointer-events:none;animation:paint 1s infinite alternate}',expected:false},
  ];
  for (const test of cases) {
    await page.setContent(`<style>body{margin:0}.wrap{position:relative;padding:40px}h1{font:40px Arial;margin:0}${test.css||''}</style>${test.html}`);
    const before=await page.locator('#target').evaluate(el=>el.outerHTML);
    let actual=false,probe,error;
    try {probe=await visiblePixelContribution(page,'#target',out,test.name);actual=true;}
    catch(e){probe=e.probe;error=String(e.message||e);}
    const restored=before===await page.locator('#target').evaluate(el=>el.outerHTML);
    const passed=actual===test.expected && restored;
    results.push({name:test.name,passed,expected:test.expected,actual,stylesRestored:restored,probe,error});
  }
  console.log(JSON.stringify({passed:results.every(x=>x.passed),results,output:out}));
  process.exitCode=results.every(x=>x.passed)?0:1;
} finally {await browser.close();}
