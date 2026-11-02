"""Build an immutable private Kaggle diagnostic notebook without credentials."""
import ast
import hashlib
import json
from pathlib import Path
import re

from .kaggle import prepare,verify_payload
from .assets import pinned_datasets


OLD_COMMAND="""    args=[sys.executable,'-u','-m','reproduce.run','--task',TASK,'--mode',MODE,'--out',str(OUT)]
    if RESUME_DATASET:
        matches=[p for p in pathlib.Path('/kaggle/input').rglob('last.pt') if hashlib.sha256(p.read_bytes()).hexdigest()==RESUME_SHA]
        if len(matches)!=1:raise ValueError('resume checkpoint SHA missing or ambiguous')
        args+=['--resume',str(matches[0])]
"""

NEW_COMMAND="""    owner,state_slug,state_version=STATE_DATASET.split('/')
    roots=[pathlib.Path('/kaggle/input')/state_slug,pathlib.Path('/kaggle/input/datasets')/owner/state_slug]
    matches=sorted({(p/'last.pt').resolve() for p in roots if (p/'last.pt').is_file()})
    if len(matches)!=1 or hashlib.sha256(matches[0].read_bytes()).hexdigest()!=RESUME_SHA:
        raise ValueError('owned checkpoint capsule missing or SHA differs')
    args=[sys.executable,'-u','-m','reproduce.proxy_recovery','--task',TASK,
          '--checkpoint',str(matches[0]),'--checkpoint-sha256',RESUME_SHA,
          '--training-source-sha256',DIAGNOSTIC['training_source_sha256'],
          '--stage',DIAGNOSTIC['stage'],'--max-seconds',str(DIAGNOSTIC['max_seconds']),
          '--out',str(OUT)]
"""


def prepare_diagnostic(root,directory,account,slug,task,dataset,checkpoint_sha,training_source,
                       stage='diagnose',max_seconds=7200):
    if (task not in ('ot','ar') or stage not in ('diagnose','recover','evaluate')
        or max_seconds<=0 or max_seconds>(7200 if stage=='diagnose' else 18000)):
        raise ValueError('invalid diagnostic stage or budget')
    if not re.fullmatch(re.escape(account)+r'/[a-z0-9-]+/[1-9][0-9]*',dataset):
        raise ValueError('owned version-pinned private state required')
    if not all(re.fullmatch(r'[0-9a-f]{64}',s) for s in (checkpoint_sha,training_source)):
        raise ValueError('exact checkpoint and legacy source SHA required')
    directory=Path(directory)
    job=prepare(root,directory,account,slug,task,'full','/'.join(dataset.split('/')[:2]),checkpoint_sha)
    notebook=json.loads((directory/'notebook.ipynb').read_text())
    code=''.join(notebook['cells'][0]['source'])
    if code.count(OLD_COMMAND)!=1:raise ValueError('canonical bootstrap command changed')
    settings=dict(stage=stage,max_seconds=max_seconds,training_source_sha256=training_source)
    code=f'DIAGNOSTIC={settings!r}\nSTATE_DATASET={dataset!r}\n'+code.replace(OLD_COMMAND,NEW_COMMAND)
    compile(code,'diagnostic-bootstrap','exec')
    notebook['cells'][0]['source']=code.splitlines(True)
    meta=json.loads((directory/'kernel-metadata.json').read_text())
    meta['dataset_sources']=pinned_datasets(task)+[dataset]
    job.update(resume_dataset=dataset,diagnostic=settings,private_checkpoint_source=dataset)
    for name,value in [('notebook.ipynb',notebook),('kernel-metadata.json',meta)]:
        (directory/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    job['payload_hashes']={name:hashlib.sha256((directory/name).read_bytes()).hexdigest()
        for name in ('notebook.ipynb','kernel-metadata.json')}
    (directory/'job.json').write_text(json.dumps(job,indent=2))
    verify_diagnostic(directory,root)
    return job


def verify_diagnostic(directory,root=None):
    directory=Path(directory);meta,job=verify_payload(directory,root)
    source=''.join(json.loads((directory/'notebook.ipynb').read_text())['cells'][0]['source'])
    constants={}
    for statement in ast.parse(source).body:
        if isinstance(statement,ast.Assign) and len(statement.targets)==1 and isinstance(statement.targets[0],ast.Name):
            name=statement.targets[0].id
            if name in ('DIAGNOSTIC','STATE_DATASET','RESUME_SHA'):
                constants[name]=ast.literal_eval(statement.value)
    if (constants.get('DIAGNOSTIC')!=job['diagnostic'] or constants.get('STATE_DATASET')!=job['resume_dataset']
        or constants.get('RESUME_SHA')!=job['resume_sha256'] or OLD_COMMAND in source or NEW_COMMAND not in source
        or job['resume_dataset'].split('/')[0]!=meta['id'].split('/')[0]):
        raise ValueError('diagnostic command or private checkpoint binding differs')
    compile(source,'verified-diagnostic','exec')
    return meta,job
