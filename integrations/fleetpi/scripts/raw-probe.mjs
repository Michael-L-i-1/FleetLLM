#!/usr/bin/env node
import {readFileSync,writeFileSync} from 'node:fs';
import {resolve,join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {EngineSupervisor,vllmServerSpec} from '../vendor/pi/packages/fleetpi/dist/index.js';
const root=fileURLToPath(new URL('../../../',import.meta.url));const out=resolve(process.argv[2]);
const spec=vllmServerSpec({repoRoot:root,configPath:join(root,'integrations/fleetpi/configs/vllm-default.local.json'),logPath:join(out,'engine.log')});
const supervisor=new EngineSupervisor(spec,(state,data)=>console.log(JSON.stringify({state,...data})));
try{await supervisor.start();
 const req=JSON.parse(readFileSync(join(out,'request.json'),'utf8'));
 const ids=JSON.parse(readFileSync(join(out,'token-ids.json'),'utf8'));
 const raw=await fetch(`${spec.baseUrl}/completions`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:'glimmer-mlx4',prompt:ids,temperature:0,max_tokens:512,stream:false,skip_special_tokens:false,return_token_ids:true})}).then(r=>r.json());
 writeFileSync(join(out,'raw-completion.json'),JSON.stringify(raw,null,2)+'\n');console.log(JSON.stringify({raw}));
 for(const stream of [false,true]){
  const response=await fetch(`${spec.baseUrl}/chat/completions`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...req,stream,stream_options:stream?req.stream_options:undefined})});
  const text=await response.text();writeFileSync(join(out,stream?'parsed-stream.sse':'parsed-nonstream.json'),text);console.log(JSON.stringify({stream,status:response.status,last:text.slice(-300)}));
 }
}finally{await supervisor.stop();}
