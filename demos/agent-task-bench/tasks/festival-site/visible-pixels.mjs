// Detect a target's actual rendered contribution, including non-hit-testable occlusion.
// This uses Playwright's bundled PNG decoder; no image dependency is installed.
import {resolve} from 'node:path';
import {writeFile} from 'node:fs/promises';
import {PNG} from './dependencies.mjs';

function delta(a, b, index) {
  return Math.max(...[0,1,2].map(c => Math.abs(a.data[index+c]-b.data[index+c])));
}

export async function visiblePixelContribution(page, selector, output, name) {
  const target = page.locator(selector);
  if (await target.count() !== 1) throw Error(`Expected exactly one target for ${selector}`);
  if (!await target.isVisible()) throw Error(`${selector} has no visible layout box`);
  await target.scrollIntoViewIfNeeded();
  await page.evaluate(() => document.fonts.ready);
  // Pause real animations at their present frame. Disable transitions only during
  // the probe, so visibility changes cannot themselves animate. Restore both.
  const motion = await page.evaluateHandle(async () => {
    const states = document.getAnimations().map(animation => ({animation, playState:animation.playState, currentTime:animation.currentTime}));
    for (const state of states) state.animation.pause();
    await Promise.all(states.map(state => state.animation.ready.catch(() => {})));
    return states;
  });
  const transitionGuard = await page.addStyleTag({content:'*,*::before,*::after{transition:none!important}'});
  const styles = await target.evaluateHandle(root => [root,...root.querySelectorAll('*')].map(el => ({el,style:el.getAttribute('style')})));
  const restoreStyles = () => page.evaluate(states => {
    for (const {el,style} of states) style===null ? el.removeAttribute('style') : el.setAttribute('style',style);
  },styles);
  try {
    const rect = await target.boundingBox();
    const viewport = page.viewportSize();
    if (!rect) throw Error(`${selector} has no bounding box`);
    const x = Math.max(0, Math.floor(rect.x)), y = Math.max(0, Math.floor(rect.y));
    const right = Math.min(viewport.width, Math.ceil(rect.x+rect.width));
    const bottom = Math.min(viewport.height, Math.ceil(rect.y+rect.height));
    const clip = {x,y,width:right-x,height:bottom-y};
    if (clip.width<2 || clip.height<2) throw Error(`${selector} is outside the viewport`);
    const shot = () => page.screenshot({clip, caret:'hide', scale:'css'});
    // Two unchanged controls and one restored control exclude moving pixels from
    // the contribution count. Hiding changes visibility, not element geometry.
    const beforeA = await shot(), beforeB = await shot();
    await page.evaluate(states => { for (const {el} of states) el.style.setProperty('visibility','hidden','important'); },styles);
    if (await target.evaluate(el => getComputedStyle(el).visibility) !== 'hidden') throw Error('Probe could not hide target');
    const hidden = await shot();
    await restoreStyles();
    const restored = await shot();
    if (!await page.evaluate(states => states.every(({el,style}) => el.getAttribute('style')===style),styles)) throw Error('Original inline styles were not restored');
    const [a,b,h,r] = [beforeA,beforeB,hidden,restored].map(buf => PNG.sync.read(buf));
    if (![b,h,r].every(img => img.width===a.width && img.height===a.height)) throw Error('Probe screenshot geometry changed');
    let stableChangedPixels=0, unstablePixels=0, changedPixels=0;
    const channelThreshold=8;
    for (let i=0;i<a.data.length;i+=4) {
      const unstable=delta(a,b,i)>=channelThreshold || delta(b,r,i)>=channelThreshold;
      const changed=delta(b,h,i)>=channelThreshold;
      if (unstable) unstablePixels++;
      if (changed) changedPixels++;
      if (changed && !unstable) stableChangedPixels++;
    }
    // A tiny nonzero threshold rejects screenshot noise; ordinary visible text
    // exceeds it comfortably without prescribing a font, color, size or layout.
    const minimumChangedPixels=24;
    const result={selector, clip, stableChangedPixels, changedPixels, unstablePixels, minimumChangedPixels, channelThreshold,
      passed:stableChangedPixels>=minimumChangedPixels, styleRestored:true};
    await Promise.all([
      writeFile(resolve(output,`${name}-before.png`),beforeB),
      writeFile(resolve(output,`${name}-hidden.png`),hidden),
      writeFile(resolve(output,`${name}-restored.png`),restored),
    ]);
    if (!result.passed) throw Object.assign(Error(`${selector} contributes only ${stableChangedPixels} stable changed pixels; requires at least ${minimumChangedPixels} (possible occlusion or invisible content)`),{probe:result});
    return result;
  } finally {
    await restoreStyles();
    await transitionGuard.evaluate(el => el.remove());
    await page.evaluate(states => {
      for (const state of states) {
        try {
          state.animation.currentTime=state.currentTime;
          if (state.playState==='running') state.animation.play();
          else if (state.playState==='finished') state.animation.finish();
          else if (state.playState==='idle') state.animation.cancel();
          else state.animation.pause();
        } catch {}
      }
    },motion);
    await motion.dispose();
    await styles.dispose();
  }
}
