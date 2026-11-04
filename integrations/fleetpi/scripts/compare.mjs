#!/usr/bin/env node
// Serialized real-Pi demo; captures actual workspace snapshots and owned process memory.
import {spawn} from 'node:child_process';
import {mkdirSync, openSync, closeSync, readFileSync, writeFileSync, existsSync} from 'node:fs';
import {join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {once} from 'node:events';
import {EngineSupervisor, fleetServerSpec, vllmServerSpec, MAC_M5PRO_48GB_GLIMMER, runFestival, DEFAULT_LIMITS} from '../vendor/pi/packages/fleetpi/dist/index.js';
const root=fileURLToPath(new URL('../../../',import.meta.url));
const output=resolve(process.argv[2]??join(root,'results/fleetpi/comparison'));
const modes=process.argv.slice(3);if(!modes.length)modes.push('fleet8','vllm8','vllm1');
if(modes.some(x=>!['fleet8','vllm8','vllm1'].includes(x)))throw Error('Unknown mode');
mkdirSync(output,{recursive:true});
const task=join(root,'demos/agent-task-bench/tasks/festival-site');
const cancel=new AbortController();let supervisor;let interruptions=0;
for(const signal of ['SIGINT','SIGTERM'])process.on(signal,()=>{if(++interruptions===1)cancel.abort();else{supervisor?.signalSync('SIGKILL');process.exit(130);}});
const pause=ms=>new Promise(r=>setTimeout(r,ms));
function start(command,args,log){const fd=openSync(log,'a');try{return spawn(command,args,{cwd:root,stdio:['ignore',fd,fd]});}finally{closeSync(fd);}}
async function stop(child){if(!child||child.exitCode!==null)return;const exited=once(child,'exit');child.kill('SIGTERM');await Promise.race([exited,pause(15000)]);if(child.exitCode===null){child.kill('SIGKILL');await exited;}}
for(const mode of modes){
 if(cancel.signal.aborted)break;
 const caseDir=join(output,mode); if(existsSync(caseDir))throw Error(`Refusing existing case ${caseDir}`);
 mkdirSync(caseDir,{recursive:true});const runDir=join(caseDir,'run');const capture=join(caseDir,'capture');
 const spec=mode==='fleet8'?fleetServerSpec({repoRoot:root,profile:MAC_M5PRO_48GB_GLIMMER,port:8090,logPath:join(caseDir,'engine.log'),fleetBin:join(root,'build-fleetpi/fleet-serve')}):vllmServerSpec({repoRoot:root,configPath:join(root,'integrations/fleetpi/configs/vllm-default.local.json'),logPath:join(caseDir,'engine.log')});
 writeFileSync(join(caseDir,'config.json'),JSON.stringify({mode,server:spec,limits:DEFAULT_LIMITS,note:'Same real Pi core/tools/checker/task/target weights; Fleet extension only in fleet8. Stock vLLM scheduling/cache defaults plus required Glimmer correctness overlay and parsers.'},null,2)+'\n');
 supervisor=new EngineSupervisor(spec,(state,detail)=>console.log(JSON.stringify({mode,event:'engine',state,...detail})));
 let recorder,engineMemory,wholeMemory;const captureStop=join(caseDir,'capture.stop');
 try{
  await supervisor.start(cancel.signal);
  engineMemory=start('python3',[join(root,'demos/agent-task-bench/sample_memory.py'),String(supervisor.pid),join(caseDir,'engine-memory.csv')],join(caseDir,'sampler.log'));
  wholeMemory=start('python3',[join(root,'demos/agent-task-bench/sample_memory.py'),String(process.pid),join(caseDir,'system-memory.csv')],join(caseDir,'sampler.log'));
  const startUnix=Date.now()/1000;
  const result=await runFestival({mode,taskDir:task,outputDir:runDir,baseUrl:spec.baseUrl,model:'glimmer-mlx4',signal:cancel.signal,startTimeUnix:startUnix,
   beforeRun:async({workspace})=>{
    recorder=start('node',[join(root,'demos/agent-task-bench/visual/record.mjs'),'--workspace',workspace,'--output',capture,'--events',join(runDir,'events.jsonl'),'--stop-file',captureStop,'--start-time-unix',String(startUnix),'--renderer',join(task,'render.py'),'--pages','index.html,lineup.html,schedule.html,venues.html,tickets.html,accessibility.html,volunteer.html,faq.html','--entry','index.html'],join(caseDir,'capture.log'));
    const deadline=Date.now()+60000;
    while(!existsSync(join(capture,'capture-ready.json'))){if(recorder.exitCode!==null)throw Error('Recorder exited before initial frame');if(Date.now()>deadline)throw Error('Recorder initial frame timeout');await pause(250);}
    if(JSON.parse(readFileSync(join(capture,'capture-ready.json'),'utf8')).frames<1)throw Error('No actual starter screenshot');
   },
   onEvent:e=>{if(['round_start','round_complete','run_end','agent_error','infrastructure_error'].includes(e.event))console.log(JSON.stringify(e));},
  });
  writeFileSync(join(caseDir,'result.json'),JSON.stringify(result,null,2)+'\n');
  writeFileSync(join(caseDir,'timing.json'),JSON.stringify({captureStartUnix:startUnix,finishedUnix:Date.now()/1000,taskWallSeconds:result.totals.wallSeconds,note:'Task wall excludes server startup and initial capture hook. Video clock includes initial capture setup; sampled equally in every lane.'},null,2)+'\n');
  if(result.infrastructureError)throw Error(result.infrastructureError);
 }finally{
  writeFileSync(captureStop,'stop\n');
  if(recorder&&recorder.exitCode===null){await Promise.race([once(recorder,'exit'),pause(30000)]);await stop(recorder);}
  await stop(engineMemory);await stop(wholeMemory);await supervisor.stop();supervisor=undefined;
 }
}
