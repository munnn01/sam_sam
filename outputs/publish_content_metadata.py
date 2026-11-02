import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'outputs/content_rate_20261101'
registry=json.loads((ROOT/'CONTENT_RATE_RUNS_20261101.json').read_text())
registry['verification']=json.loads((BASE/'verification.json').read_text())
registry['code_commit']=registry['verification']['source_commit']
(ROOT/'CONTENT_RATE_RUNS_20261101.json').write_text(json.dumps(registry,indent=2))
(BASE/'deployment_receipt.json').write_text(json.dumps(registry,indent=2))
paths={'paper_reproduction/CONTENT_RATE_RUNS_20261101.json':ROOT/'CONTENT_RATE_RUNS_20261101.json'}
elements=[];proof={}
for name,path in paths.items():
 raw=path.read_bytes();elements.append(dict(path=name,mode='100644',type='blob',content=raw.decode('utf-8')))
 proof[name]=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
print(json.dumps(dict(elements=elements,proof=proof),ensure_ascii=True))
