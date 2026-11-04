#!/usr/bin/env node
// Export the exact active Pi tool definitions used by the benchmark; no model request.
import {mkdtempSync, rmSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createLocalModelRuntime, PathGuard, commonToolsExtension} from '../vendor/pi/packages/fleetpi/dist/index.js';
import {openSession} from '../vendor/pi/packages/fleetpi/dist/session/open-session.js';
const dir=mkdtempSync(join(tmpdir(),'fleetpi-tools-'));
let session;
try{
 const {runtime,model}=await createLocalModelRuntime(join(dir,'agent'),'http://127.0.0.1:1/v1','glimmer-mlx4',{contextWindow:96000,maxOutputTokens:4096});
 const guard=new PathGuard(dir,['index.html']);
 session=await openSession({cwd:dir,agentDir:join(dir,'agent'),sessionDir:join(dir,'sessions'),modelRuntime:runtime,model,extensions:[commonToolsExtension(dir,guard,{ownedFiles:['index.html'],currentRound:()=>1,check:async()=>{throw Error('No model calls');}})]});
 const tools=session.agent.state.tools.map(({name,description,parameters})=>({type:'function',function:{name,description,parameters}}));
 writeFileSync(process.argv[2],JSON.stringify(tools,null,2)+'\n');
}finally{if(session)await session.dispose();rmSync(dir,{recursive:true,force:true});}
