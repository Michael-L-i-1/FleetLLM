#!/usr/bin/env node
// Real Pi sessions, official Glimmer tool calls, two concurrent edit/read round trips.
import {cpSync, mkdirSync, readFileSync, writeFileSync} from 'node:fs';
import {resolve, join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {EngineSupervisor, fleetServerSpec, vllmServerSpec, MAC_M5PRO_48GB_GLIMMER, createLocalModelRuntime, PathGuard, commonToolsExtension} from '../vendor/pi/packages/fleetpi/dist/index.js';
import {openSession} from '../vendor/pi/packages/fleetpi/dist/session/open-session.js';
const root=fileURLToPath(new URL('../../../',import.meta.url));
const engine=process.argv[2];
if(!['fleet','vllm'].includes(engine)) throw Error('Usage: smoke.mjs fleet|vllm OUTPUT');
const out=resolve(process.argv[3]); mkdirSync(join(out,'logs'),{recursive:true});
const workspace=join(out,'workspace');
cpSync(join(root,'demos/agent-task-bench/tasks/festival-site/fixture'),workspace,{recursive:true});
const spec=engine==='fleet'?fleetServerSpec({repoRoot:root,profile:MAC_M5PRO_48GB_GLIMMER,port:8090,logPath:join(out,'logs/engine.log'),fleetBin:join(root,'build-fleetpi/fleet-serve')}):vllmServerSpec({repoRoot:root,configPath:join(root,'integrations/fleetpi/configs/vllm-default.local.json'),logPath:join(out,'logs/engine.log')});
const supervisor=new EngineSupervisor(spec,(state,data)=>console.log(JSON.stringify({state,...data})));
const cancel=new AbortController(); const timer=setTimeout(()=>cancel.abort(),240000);
const sessions=[]; const events=[];
process.on('SIGINT',()=>cancel.abort());process.on('SIGTERM',()=>cancel.abort());
try{
 await supervisor.start(cancel.signal);
 const {runtime,model}=await createLocalModelRuntime(join(out,'pi-agent'),spec.baseUrl,'glimmer-mlx4',{contextWindow:96000,maxOutputTokens:2048});
 await Promise.all(['home','faq'].map(async(page)=>{
  const file=`pages/${page}.html`; const original=readFileSync(join(workspace,file),'utf8');
  const before=page==='home'?'An arts weekend in the park.':'Answers coming soon.';
  const after=`Verified Pi ${page} & café <strong>ready</strong>.`;
  const guard=new PathGuard(workspace,[file]);
  const session=await openSession({cwd:workspace,agentDir:join(out,'pi-agent'),sessionDir:join(out,'sessions',page),modelRuntime:runtime,model,extensions:[commonToolsExtension(workspace,guard,{ownedFiles:[file],currentRound:()=>1,check:async()=>{throw Error('Not part of smoke');}})]});
  sessions.push(session);cancel.signal.addEventListener('abort',()=>void session.abort(),{once:true});
  session.subscribe(e=>{events.push({page,...e});if(['tool_execution_start','tool_execution_end'].includes(e.type))console.log(JSON.stringify({page,type:e.type,tool:e.toolName,isError:e.isError}));});
  await session.prompt(`This is an actual tool smoke test. Use read to inspect ${file}, then edit to replace exactly the text ${JSON.stringify(before)} with ${JSON.stringify(after)}. Preserve everything else byte for byte. Then use read to verify the edit and give a brief confirmation. Do not run check_site. Your sole writable file is ${file}.`);
  const actual=readFileSync(join(workspace,file),'utf8');
  if(actual!==original.replace(before,after))throw Error(`${page}: exact edit mismatch`);
  await session.prompt(`Use read to inspect ${file} once more. Report whether the exact phrase ${JSON.stringify(after)} is present. Do not edit anything.`);
  const tools=events.filter(e=>e.page===page&&e.type==='tool_execution_start').map(e=>e.toolName);
  if(!tools.includes('edit')||tools.filter(t=>t==='read').length<3)throw Error(`${page}: missing tool chain ${tools}`);
  console.log(JSON.stringify({page,passed:true,tools,sessionFile:session.sessionFile}));
 }));
 writeFileSync(join(out,'result.json'),JSON.stringify({passed:true,engine},null,2)+'\n');
}finally{
 clearTimeout(timer);for(const session of sessions)await session.dispose();
 writeFileSync(join(out,'events.json'),JSON.stringify(events,null,2)+'\n');
 await supervisor.stop();
}
