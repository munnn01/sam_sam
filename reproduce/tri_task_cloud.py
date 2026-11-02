"""Private immutable notebooks for separate segmentation, detection and tracking trials."""
import ast
import hashlib
import json
from pathlib import Path

from .assets import pinned_datasets
from .image_proxy_cloud import validate_legacy
from .image_study_cloud import STUDY_COMMAND
from .kaggle import prepare as native_prepare,source_files
from .proxy_cloud import OLD_COMMAND
from .tri_task_data import config,PUBLIC_PROXY

COMMAND=STUDY_COMMAND.replace("'reproduce.image_study'","'reproduce.tri_task_study'")

def prepare(root,directory,account,slug,cfg):
    if cfg!=config(cfg['task'],cfg['sam_arm']):raise ValueError('three-task recipe differs')
    directory=Path(directory);task=cfg['task'];base='ot' if task=='ot' else 'od'
    job=native_prepare(root,directory,account,slug,base)
    notebook=json.loads((directory/'notebook.ipynb').read_text());code=''.join(notebook['cells'][0]['source'])
    if code.count(OLD_COMMAND)!=1:raise ValueError('canonical bootstrap command differs')
    code=code.replace('TASK='+repr(base),'TASK='+repr(task),1)
    code=code.replace("if TASK in ('od','ic'):","if TASK in ('od','ic','is','ot'):")
    code=f'STUDY={cfg!r}\nINPUT_SPEC={PUBLIC_PROXY!r}\n'+code.replace(OLD_COMMAND,COMMAND)
    compile(code,'three-task-bootstrap','exec');notebook['cells'][0]['source']=code.splitlines(True)
    meta=json.loads((directory/'kernel-metadata.json').read_text());meta['dataset_sources']=pinned_datasets(base)+[PUBLIC_PROXY['ref']]
    job.update(task=task,config=cfg,input_spec=PUBLIC_PROXY,tri_task=True)
    for name,value in (('notebook.ipynb',notebook),('kernel-metadata.json',meta)):
        (directory/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    job['payload_hashes']={name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in ('notebook.ipynb','kernel-metadata.json')}
    (directory/'job.json').write_text(json.dumps(job,indent=2));verify(directory,root);return job

def verify(directory,root=None):
    directory=Path(directory);job=json.loads((directory/'job.json').read_text());meta=json.loads((directory/'kernel-metadata.json').read_text())
    cfg=job['config'];task=cfg['task'];base='ot' if task=='ot' else 'od'
    if cfg!=config(task,cfg['sam_arm']) or job['input_spec']!=PUBLIC_PROXY:raise ValueError('three-task recipe/input differs')
    if set(job['payload_hashes'])!={'notebook.ipynb','kernel-metadata.json'}:raise ValueError('both payload hashes required')
    for name,sha in job['payload_hashes'].items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=sha:raise ValueError('payload checksum differs')
    code=''.join(json.loads((directory/'notebook.ipynb').read_text())['cells'][0]['source']);constants={}
    names={'TASK','MODE','MANIFEST','SOURCE_SHA','BUNDLE','STUDY','INPUT_SPEC','RESUME_DATASET','RESUME_SHA'}
    for node in ast.parse(code).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in names:
            constants[node.targets[0].id]=ast.literal_eval(node.value)
    if (constants['TASK']!=task or constants['MODE']!='full' or constants['STUDY']!=cfg or constants['INPUT_SPEC']!=PUBLIC_PROXY
        or constants['RESUME_DATASET'] is not None or constants['RESUME_SHA'] is not None or constants['SOURCE_SHA']!=job['source_sha256']
        or constants['MANIFEST']!=job['source_hashes'] or meta['id']!=job['handle'] or meta['is_private'] is not True
        or meta['enable_gpu'] is not True or meta['enable_internet'] is not True
        or meta['dataset_sources']!=pinned_datasets(base)+[PUBLIC_PROXY['ref']] or meta['kernel_sources']!=[]
        or COMMAND not in code or OLD_COMMAND in code or "if TASK in ('od','ic','is','ot'):" not in code):raise ValueError('cloud binding differs')
    validate_legacy(constants['BUNDLE'],constants['MANIFEST'],constants['SOURCE_SHA'])
    if root is not None and source_files(root)!=job['source_hashes']:raise ValueError('source changed since packaging')
    compile(code,'verified-three-task-bootstrap','exec');return meta,job
