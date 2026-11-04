from pathlib import Path
import json
from transformers.utils.chat_template_utils import _compile_jinja_template
R=Path('results/fleetpi/setup/reference');template=(R/'glimmer-chat-template.jinja').read_text();compiled=_compile_jinja_template(template)
tools=[{'type':'function','function':{'name':name,'description':desc,'parameters':{'type':'object','properties':props,'required':required}}} for name,desc,props,required in [
 ('read','Read a local UTF-8 file.',{'path':{'type':'string'},'offset':{'type':'number'},'limit':{'type':'number'}},['path']),
 ('edit','Replace exact text; preserve whitespace and Unicode.',{'path':{'type':'string'},'oldText':{'type':'string'},'newText':{'type':'string'}},['path','oldText','newText']),
 ('write','Write complete file contents.',{'path':{'type':'string'},'content':{'type':'string'}},['path','content'])]]
cases=[{'name':'tool-definitions','messages':[{'role':'system','content':'You are a coding agent.'},{'role':'user','content':'Read index.html.'}],'tools':tools},
 {'name':'historical-calls-reversed-results','messages':[{'role':'system','content':'Preserve project requirements.'},{'role':'user','content':'Inspect files.'},{'role':'assistant','content':None,'reasoning_content':'I will inspect the files.','tool_calls':[{'id':'call-a','type':'function','function':{'name':'read','arguments':{'path':'index.html','offset':1}}},{'id':'call-b','type':'function','function':{'name':'edit','arguments':{'path':'index.html','oldText':'  café & <button title="Hi">\n','newText':'  café & <button title="Hello">\n'}}}]},{'role':'tool','tool_call_id':'call-b','content':'Edit applied.'},{'role':'tool','tool_call_id':'call-a','content':'<h1>Demo</h1>'},{'role':'user','content':'Continue.'}],'tools':tools},
 {'name':'adjacent-assistant-and-explicit-tool-name','messages':[{'role':'system','content':'Use local files.'},{'role':'assistant','content':'Planning.','recipient':'self','end_turn':False},{'role':'assistant','content':None,'tool_calls':[{'id':'call-c','type':'function','function':{'name':'write','arguments':{'path':'x.txt','content':'  true story\n123 Main Street\n'}}}]},{'role':'tool','name':'write','tool_call_id':'call-c','content':'Saved.'}],'tools':tools},
 {'name':'text-only-regression','messages':[{'role':'system','content':'You are helpful.'},{'role':'user','content':'Hello.'},{'role':'assistant','content':'Hi.'},{'role':'user','content':'Continue.'}],'tools':[]}]
for case in cases:
 case['render_options']={'bos_token':'<|begin_of_text|>','add_generation_prompt':True,'reasoning_strength':'low','current_date':'2026-11-04'}
 case['expected']=compiled.render(messages=case['messages'],tools=case['tools'],**case['render_options'])
(R/'template-golden-fixtures.json').write_text(json.dumps({'source':'Official cached checkpoint Jinja at3e7677d7a40d348a3daba263a2b1c0aa41910710; HF _compile_jinja_template','cases':cases},ensure_ascii=False,indent=2)+'\n')
print('Generated',len(cases),'independent official-template golden fixtures')
