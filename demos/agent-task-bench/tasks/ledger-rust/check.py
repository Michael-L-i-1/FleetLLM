#!/usr/bin/env python3
"""Protected acceptance checker. No network, package manager or mutable expected data."""
import argparse
import csv
import datetime as dt
import io
import json
from pathlib import Path
import random
import re
import subprocess
import tempfile

TASK = Path(__file__).resolve().parent
GROUPS = ['money', 'date', 'csv', 'labels', 'tags', 'filter', 'summary', 'audit']
ASCII_SPACE = ' \t\r\n\v\f'
LOWER = str.maketrans('ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')

def canonical(cents):
    return f'{"-" if cents < 0 else ""}{abs(cents)//100}.{abs(cents)%100:02}'

def oracle(text, start, end):
    rows, invalid = [], 0
    for line in text.splitlines()[1:]:
        try:
            fields = next(csv.reader([line], strict=True))
            if len(fields) != 5: raise ValueError('columns')
            ident, date, vendor, money, tags = fields
            ident = ident.strip(ASCII_SPACE)
            vendor = re.sub(r'[ \t\r\n\v\f]+', ' ', vendor.strip(ASCII_SPACE)).translate(LOWER)
            if not ident or not vendor or not re.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}', date): raise ValueError('identity/date')
            dt.date.fromisoformat(date)
            if not re.fullmatch(r'[+-]?[0-9]+(?:\.[0-9]{1,2})?', money): raise ValueError('money')
            sign = -1 if money.startswith('-') else 1
            whole, _, frac = money.lstrip('+-').partition('.')
            cents = sign*(int(whole)*100+int((frac+'00')[:2]))
            if not -(2**63) <= cents < 2**63: raise ValueError('overflow')
            for t in tags.split(';'):
                t=t.strip(ASCII_SPACE)
                if t and not re.fullmatch(r'[A-Za-z0-9_-]+',t): raise ValueError('tag')
            rows.append((ident,date,vendor,cents))
        except (ValueError,csv.Error): invalid+=1
    unique, duplicates, seen = [], set(), set()
    for row in rows:
        if row[0] in seen: duplicates.add(row[0])
        else: unique.append(row);seen.add(row[0])
    selected = [r for r in unique if start <= r[1] <= end]
    groups = {}
    for _,_,vendor,cents in selected:
        count,total=groups.get(vendor,(0,0));groups[vendor]=count+1,total+cents
    return {'accepted':len(unique),'invalid':invalid,'duplicates':sorted(duplicates),
            'selected':[r[0] for r in selected],
            'totals':[{'vendor':k,'count':groups[k][0],'total':canonical(groups[k][1])} for k in sorted(groups)]}

def generated(seed):
    rng=random.Random(seed);out=io.StringIO();w=csv.writer(out,lineterminator='\n')
    w.writerow(['id','date','vendor','amount','tags'])
    names=['Coffee Bar','  GREEN   MARKET ','Book, Shop','Café MÜNCHEN','Zero Store']
    for i in range(36):
        day=dt.date(2024,1,1)+dt.timedelta(days=rng.randrange(120))
        cents=rng.randrange(-900,2001)
        ident=f'id{rng.randrange(18):02}'
        w.writerow([ident,day.isoformat(),rng.choice(names),canonical(cents),rng.choice(['food; WORK','books;gift',';local;local',''])])
    w.writerow(['bad-date','2023-02-29','X','1.00',''])
    w.writerow(['bad-money','2024-01-01','X','1.234',''])
    w.writerow(['bad-tag','2024-01-01','X','1.00','bad tag'])
    w.writerow(['','2024-01-01','X','1.00',''])
    return out.getvalue()

def run(cmd, timeout=30):
    return subprocess.run(cmd,text=True,capture_output=True,timeout=timeout)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--output',type=Path)
    a=p.parse_args();ws=a.workspace.resolve();results=[];compile_errors=[]
    with tempfile.TemporaryDirectory(prefix='ledger-acceptance-') as directory:
        tmp=Path(directory);app=tmp/'ledger';probe=tmp/'probe';ref=tmp/'reference'
        template=(TASK/'probe.rs.template').read_text()
        for name in GROUPS:template=template.replace('__'+name.upper()+'__',json.dumps(str(ws/(name+'.rs'))))
        (tmp/'probe.rs').write_text(template)
        builds=[('application',['rustc','--edition=2021','-C','opt-level=1',str(ws/'main.rs'),'-o',str(app)]),
                ('module tests',['rustc','--edition=2021','--test','-C','opt-level=1',str(tmp/'probe.rs'),'-o',str(probe)]),
                ('C reference',['cc','-std=c11','-O1',str(TASK/'fixture/reference.c'),'-o',str(ref)])]
        for name,cmd in builds:
            try:r=run(cmd)
            except Exception as e:compile_errors.append({'stage':name,'error':str(e)});continue
            if r.returncode:compile_errors.append({'stage':name,'error':r.stderr[-6000:]})
        if not compile_errors:
            for group in GROUPS:
                try:
                    r=run([str(probe),'--exact',group,'--nocapture'],timeout=8)
                    results.append({'name':group,'passed':r.returncode==0,'detail':(r.stdout+r.stderr)[-3500:] if r.returncode else 'module API acceptance passed'})
                except Exception as e:results.append({'name':group,'passed':False,'detail':str(e)})
            cases=[('integration-fixture',(TASK/'fixture/fixtures/ledger.csv').read_text(),'2024-02-01','2024-03-31'),
                   ('integration-generated-17',generated(17),'2024-02-01','2024-03-31'),
                   ('integration-generated-91',generated(91),'2024-01-01','2024-04-30')]
            for name,text,start,end in cases:
                path=tmp/(name+'.csv');path.write_text(text);expected=oracle(text,start,end)
                try:
                    gold=run([str(ref),'report',str(path),start,end],timeout=5)
                    if gold.returncode or json.loads(gold.stdout)!=expected:raise RuntimeError('C reference differs from independent Python oracle: '+gold.stdout[-1000:])
                    r=run([str(app),'report',str(path),start,end],timeout=5)
                    if r.returncode:
                        actual={'exit':r.returncode,'stderr':r.stderr[-1000:]}
                    else:
                        try:actual=json.loads(r.stdout)
                        except json.JSONDecodeError:actual={'invalid_json':r.stdout[-1000:]}
                    results.append({'name':name,'passed':actual==expected,'expected':expected,'actual':actual})
                except Exception as e:results.append({'name':name,'passed':False,'detail':str(e)})
    passed=sum(x['passed'] for x in results);success=not compile_errors and len(results)==11 and passed==11
    result={'success':success,'passed':passed,'total':11,'compile_errors':compile_errors,'checks':results}
    if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));return 0 if success else 1

if __name__=='__main__':raise SystemExit(main())
