#!/usr/bin/env node
// Capture Pi's exact first request against a model-free local SSE endpoint.
import {createServer} from 'node:http';
import {cpSync,existsSync,mkdirSync,readdirSync,readFileSync,writeFileSync} from 'node:fs';
import {join,resolve,relative,isAbsolute} from 'node:path';
import {fileURLToPath} from 'node:url';
import {createLocalModelRuntime,PathGuard,commonToolsExtension} from '../vendor/pi/packages/fleetpi/dist/index.js';
import {openSession} from '../vendor/pi/packages/fleetpi/dist/session/open-session.js';
const root=fileURLToPath(new URL('../../../',import.meta.url));
const source=resolve(process.argv[2]);const output=resolve(process.argv[3]);mkdirSync(output,{recursive:true});
const oldEvents=readFileSync(join(source,'run/events.jsonl'),'utf8').trim().split('\n').map(JSON.parse);
const workspace=oldEvents.find(e=>e.event==='run_start').workspace;
const rel=relative(root,workspace);
if(rel.startsWith('..')||isAbsolute(rel))throw Error('Recorded workspace is outside this checkout; use an isolated matching checkout');
if(existsSync(workspace)&&readdirSync(workspace).length)throw Error('Refusing to overwrite an existing recorded workspace; use an isolated checkout');
mkdirSync(workspace,{recursive:true});cpSync(join(source,'run/workspace'),workspace,{recursive:true});
const sessions=join(source,'run/sessions/site');
const entries=readFileSync(join(sessions,readdirSync(sessions).find(f=>f.endsWith('.jsonl'))),'utf8').trim().split('\n').map(JSON.parse);
const prompt=entries.find(e=>e.type==='message'&&e.message.role==='user').message.content.map(c=>c.text??'').join('');
const server=createServer(async(req,res)=>{let body='';for await(const chunk of req)body+=chunk;
 writeFileSync(join(output,'request.json'),JSON.stringify(JSON.parse(body),null,2)+'\n');
 res.writeHead(200,{'Content-Type':'text/event-stream'});
 for(const part of [{choices:[{index:0,delta:{role:'assistant',content:'captured'},finish_reason:null}]},{choices:[{index:0,delta:{},finish_reason:'stop'}],usage:{prompt_tokens:1,completion_tokens:1,total_tokens:2}}])res.write(`data: ${JSON.stringify({id:'capture',object:'chat.completion.chunk',model:'glimmer-mlx4',...part})}\n\n`);
 res.end('data: [DONE]\n\n');
});await new Promise(r=>server.listen(18097,'127.0.0.1',r));let session;
try{const {runtime,model}=await createLocalModelRuntime(join(output,'pi-agent'),'http://127.0.0.1:18097/v1','glimmer-mlx4',{contextWindow:96000,maxOutputTokens:4096});
 const files=['home','lineup','schedule','venues','tickets','accessibility','volunteer','faq'].map(p=>`pages/${p}.html`);
 session=await openSession({cwd:workspace,agentDir:join(output,'pi-agent'),sessionDir:join(output,'session'),modelRuntime:runtime,model,extensions:[commonToolsExtension(workspace,new PathGuard(workspace,files),{ownedFiles:files,currentRound:()=>1,check:async()=>{throw Error('Model-freecapture');}})]});
 await session.prompt(prompt);
}finally{if(session)await session.dispose();await new Promise(r=>server.close(r));}
