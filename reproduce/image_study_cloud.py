"""Immutable private image-study notebooks with exact source/config/input binding."""
import ast
import hashlib
import json
from pathlib import Path,PurePosixPath
import re

from .assets import pinned_datasets
from .image_study import expected_config
from .kaggle import prepare,source_files
from .proxy_cloud import OLD_COMMAND
from .image_proxy_cloud import validate_legacy


STUDY_COMMAND="""    configuration=OUT/'study_config.json';configuration.write_text(json.dumps(STUDY))
    args=[sys.executable,'-u','-m','reproduce.image_study','--config',str(configuration),'--out',str(OUT)]
    if INPUT_SPEC:
        owner,input_slug,input_version=INPUT_SPEC['ref'].split('/')
        roots=[pathlib.Path('/kaggle/input')/input_slug]
        if INPUT_SPEC['kind']=='dataset':roots+=[pathlib.Path('/kaggle/input/datasets')/owner/input_slug]
        else:roots+=[pathlib.Path('/kaggle/input/notebooks')/owner/input_slug,pathlib.Path('/kaggle/input/kernels')/owner/input_slug]
        paths=set()
        for root in roots:
            for base in (root,root/input_version):
                path=base/INPUT_SPEC['member']
                if path.is_file():paths.add(path.resolve())
        def stream_sha(path):
            h=hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
            return h.hexdigest()
        matches=[p for p in paths if stream_sha(p)==INPUT_SPEC['sha256']]
        if len(matches)!=1:raise ValueError('exact owned proxy input missing or ambiguous')
        args+=['--proxy',str(matches[0]),'--proxy-sha256',INPUT_SPEC['sha256']]
        print(json.dumps({'stage':'study_input_verified','task':TASK,'sha256':INPUT_SPEC['sha256']}),flush=True)
"""


def input_valid(account,cfg,spec):
    if cfg!=expected_config(cfg):raise ValueError('study configuration differs')
    if cfg['study_profile']=='proxy':
        if spec is not None:raise ValueError('Flickr proxy starts from pretrained bank, not task checkpoint')
        return
    keys={'kind','ref','member','sha256'}
    if not isinstance(spec,dict) or set(spec) not in (keys,keys|{'visibility'}) or spec['kind'] not in ('dataset','kernel'):
        raise ValueError('exact proxy input spec required')
    public=spec.get('visibility')=='public'
    if 'visibility' in spec and (not public or spec['kind']!='dataset'):raise ValueError('only datasets can be explicitly public inputs')
    owner=r'[A-Za-z0-9_-]+' if public else re.escape(account)
    if not re.fullmatch(owner+r'/[a-z0-9-]+/[1-9][0-9]*',spec['ref']):raise ValueError('owned or explicitly public version-pinned input required')
    path=PurePosixPath(spec['member'])
    if path.is_absolute() or '..' in path.parts or '\\' in spec['member'] or ':' in spec['member'] or path.suffix!='.pt':
        raise ValueError('safe relative checkpoint member required')
    if not re.fullmatch(r'[0-9a-f]{64}',spec['sha256']):raise ValueError('exact proxy SHA required')


def prepare_study(root,directory,account,slug,cfg,input_spec):
    input_valid(account,cfg,input_spec);directory=Path(directory);task=cfg['study_task']
    job=prepare(root,directory,account,slug,'ic' if task=='proxy' else task)
    notebook=json.loads((directory/'notebook.ipynb').read_text());code=''.join(notebook['cells'][0]['source'])
    if code.count(OLD_COMMAND)!=1:raise ValueError('canonical bootstrap command changed')
    if task=='proxy':
        code=code.replace("TASK='ic'","TASK='proxy'",1).replace("if TASK in ('od','ic'):","if TASK in ('od','ic','proxy'):")
    code=f'STUDY={cfg!r}\nINPUT_SPEC={input_spec!r}\n'+code.replace(OLD_COMMAND,STUDY_COMMAND)
    compile(code,'image-study-bootstrap','exec');notebook['cells'][0]['source']=code.splitlines(True)
    meta=json.loads((directory/'kernel-metadata.json').read_text())
    meta['dataset_sources']=['eeshawn/flickr30k/1'] if task=='proxy' else pinned_datasets(task)
    meta['kernel_sources']=[]
    if input_spec:
        meta['dataset_sources' if input_spec['kind']=='dataset' else 'kernel_sources'].append(input_spec['ref'])
    job.update(task=task,config=cfg,input_spec=input_spec,study=True)
    for name,value in [('notebook.ipynb',notebook),('kernel-metadata.json',meta)]:
        (directory/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    job['payload_hashes']={name:hashlib.sha256((directory/name).read_bytes()).hexdigest()
        for name in ('notebook.ipynb','kernel-metadata.json')}
    (directory/'job.json').write_text(json.dumps(job,indent=2));verify_study(directory,root);return job


def verify_study(directory,root=None):
    directory=Path(directory);job=json.loads((directory/'job.json').read_text());meta=json.loads((directory/'kernel-metadata.json').read_text())
    input_valid(job['handle'].split('/')[0],job['config'],job['input_spec'])
    if set(job['payload_hashes'])!={'notebook.ipynb','kernel-metadata.json'}:raise ValueError('both payload hashes required')
    for name,sha in job['payload_hashes'].items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=sha:raise ValueError('payload checksum differs')
    source=''.join(json.loads((directory/'notebook.ipynb').read_text())['cells'][0]['source']);constants={}
    names={'STUDY','INPUT_SPEC','TASK','MODE','SOURCE_SHA','MANIFEST','BUNDLE','RESUME_DATASET','RESUME_SHA'}
    for node in ast.parse(source).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in names:
            constants[node.targets[0].id]=ast.literal_eval(node.value)
    expected=['eeshawn/flickr30k/1'] if job['task']=='proxy' else pinned_datasets(job['task']);kernels=[]
    spec=job['input_spec']
    if spec:(expected if spec['kind']=='dataset' else kernels).append(spec['ref'])
    if (constants['STUDY']!=job['config'] or constants['INPUT_SPEC']!=spec or constants['TASK']!=job['task']
        or constants['MODE']!='full' or constants['RESUME_DATASET'] is not None or constants['RESUME_SHA'] is not None
        or constants['SOURCE_SHA']!=job['source_sha256'] or constants['MANIFEST']!=job['source_hashes']
        or meta['id']!=job['handle'] or meta['is_private'] is not True or meta['enable_gpu'] is not True
        or meta['enable_internet'] is not True or meta['dataset_sources']!=expected or meta['kernel_sources']!=kernels
        or OLD_COMMAND in source or STUDY_COMMAND not in source):raise ValueError('study binding differs')
    validate_legacy(constants['BUNDLE'],constants['MANIFEST'],constants['SOURCE_SHA'])
    if root is not None and source_files(root)!=job['source_hashes']:raise ValueError('study source changed')
    compile(source,'verified-study-bootstrap','exec');return meta,job
