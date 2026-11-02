"""Immutable private GPU notebooks; credentials remain outside this project."""
import base64
import hashlib
import json
from pathlib import Path
import re
import zlib
from .protocol import recipe
from .assets import DATASETS,pinned_datasets

BOOTSTRAP=r'''
import base64,hashlib,json,os,pathlib,subprocess,sys,tarfile,time,zlib
ROOT=pathlib.Path('/kaggle/working/paper_reproduction')
OUT=pathlib.Path('/kaggle/working/paper_results');OUT.mkdir(parents=True,exist_ok=True)
contents=json.loads(zlib.decompress(base64.b64decode(BUNDLE)))
for name,content in contents.items():
    path=ROOT/name
    if ROOT.resolve() not in path.resolve().parents:raise ValueError('source outside project')
    if hashlib.sha256(content.encode()).hexdigest()!=MANIFEST[name]:raise ValueError('source SHA mismatch')
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content,encoding='utf-8')
source_sha=hashlib.sha256(json.dumps(MANIFEST,sort_keys=True,separators=(',',':')).encode()).hexdigest()
if source_sha!=SOURCE_SHA:raise ValueError('manifest checksum differs')
print(json.dumps({'stage':'payload_verified','task':TASK,'mode':MODE,'source_sha256':source_sha}),flush=True)
environment=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                 MPLBACKEND='Agg',PAPER_SOURCE_SHA256=source_sha,PYTHONPATH=str(ROOT),MAX_JOBS='2',TORCH_CUDA_ARCH_LIST='7.5')
receipt={'task':TASK,'mode':MODE,'source_sha256':source_sha,'bootstrap_complete':False,'started':time.time()}
try:
    packages=['pycocotools>=2.0.7']
    if TASK in ('od','ic'):packages+=['compressai==1.2.8']
    if TASK=='ot':packages+=['gdown','jpeg4py','visdom','ninja','timm']
    with (OUT/'dependency_install.log').open('w') as log:
        p=subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check',*packages],stdout=log,stderr=subprocess.STDOUT,env=environment)
    if p.returncode:raise RuntimeError('dependency install failed; inspect dependency_install.log')
    print(json.dumps({'stage':'dependencies_ready','task':TASK}),flush=True)
    args=[sys.executable,'-u','-m','reproduce.run','--task',TASK,'--mode',MODE,'--out',str(OUT)]
    if RESUME_DATASET:
        matches=[p for p in pathlib.Path('/kaggle/input').rglob('last.pt') if hashlib.sha256(p.read_bytes()).hexdigest()==RESUME_SHA]
        if len(matches)!=1:raise ValueError('resume checkpoint SHA missing or ambiguous')
        args+=['--resume',str(matches[0])]
    with (OUT/'console.log').open('w') as log:
        p=subprocess.Popen(args,cwd=ROOT,env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in p.stdout:print(line,end='',flush=True);log.write(line);log.flush()
        code=p.wait()
    receipt['worker_exit_code']=code
    if code:raise RuntimeError('paper runner failed; inspect run.json and console.log')
    receipt['bootstrap_complete']=True
except BaseException as error:
    receipt['error']=f'{type(error).__name__}: {error}';raise
finally:
    receipt['wall_seconds']=time.time()-receipt['started']
    (OUT/'cloud_status.json').write_text(json.dumps(receipt,indent=2))
    with tarfile.open('/kaggle/working/paper_results.tgz','w:gz',compresslevel=1) as a:a.add(OUT,arcname='paper_results')
    print(json.dumps({'stage':'cloud_end',**receipt}),flush=True)
'''

def source_files(root):
    result={};root=Path(root)
    for p in sorted(root.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(root)
        if any(v.startswith('.') or v in ('__pycache__','outputs','build') for v in rel.parts):continue
        if p.suffix in ('.py','.toml','.md') or p.name=='LICENSE':
            result[rel.as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    return result

def prepare(root,directory,account,slug,task,mode='full',resume_dataset=None,resume_sha=None):
    if not re.fullmatch(r'[A-Za-z0-9_-]+',account) or not re.fullmatch(r'[a-z0-9][a-z0-9-]{5,70}',slug):
        raise ValueError('invalid Kaggle account/slug')
    cfg=recipe(task,mode);root=Path(root);directory=Path(directory);hashes=source_files(root)
    if 'reproduce/run.py' not in hashes:raise ValueError('complete reproduction sources required')
    source_sha=hashlib.sha256(json.dumps(hashes,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    contents={name:(root/name).read_bytes().decode('utf-8') for name in hashes}
    if resume_dataset is not None and (not re.fullmatch(re.escape(account)+r'/[a-z0-9-]+',resume_dataset)
                                       or not re.fullmatch(r'[0-9a-f]{64}',resume_sha or '')):
        raise ValueError('resume requires owned private input and exact checkpoint hash')
    bundle=base64.b64encode(zlib.compress(json.dumps(contents,ensure_ascii=False).encode(),9)).decode()
    code=(f'TASK={task!r}\nMODE={mode!r}\nMANIFEST={hashes!r}\nSOURCE_SHA={source_sha!r}\n'
          f'RESUME_DATASET={resume_dataset!r}\nRESUME_SHA={resume_sha!r}\nBUNDLE={bundle!r}\n'+BOOTSTRAP)
    compile(code,'paper-bootstrap','exec')
    notebook=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'display_name':'Python3','language':'python','name':'python3'}},
                  cells=[dict(cell_type='code',id='pinned-reproduction',metadata={},execution_count=None,outputs=[],source=code.splitlines(True))])
    metadata=dict(id=account+'/'+slug,title=slug,code_file='notebook.ipynb',language='python',kernel_type='notebook',
                  is_private=True,enable_gpu=True,enable_internet=True,dataset_sources=pinned_datasets(task)+([resume_dataset] if resume_dataset else []),
                  kernel_sources=[],competition_sources=[],model_sources=[])
    directory.mkdir(parents=True,exist_ok=False)
    for name,value in [('notebook.ipynb',notebook),('kernel-metadata.json',metadata)]:
        (directory/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    job=dict(task=task,mode=mode,handle=metadata['id'],config=cfg,source_sha256=source_sha,source_hashes=hashes,
             resume_dataset=resume_dataset,resume_sha256=resume_sha,
             payload_hashes={n:hashlib.sha256((directory/n).read_bytes()).hexdigest() for n in ('notebook.ipynb','kernel-metadata.json')})
    (directory/'job.json').write_text(json.dumps(job,indent=2));return job

def verify_payload(directory,root=None):
    directory=Path(directory);job=json.loads((directory/'job.json').read_text());meta=json.loads((directory/'kernel-metadata.json').read_text())
    for name,sha in job['payload_hashes'].items():
        if name not in ('notebook.ipynb','kernel-metadata.json') or hashlib.sha256((directory/name).read_bytes()).hexdigest()!=sha:
            raise ValueError('payload checksum differs')
    expected=pinned_datasets(job['task'])+([job['resume_dataset']] if job['resume_dataset'] else [])
    if (meta['id']!=job['handle'] or meta['is_private'] is not True or meta['enable_gpu'] is not True
        or meta['dataset_sources']!=expected or job['config']!=recipe(job['task'],job['mode'])):
        raise ValueError('private GPU task/data/config binding differs')
    if root is not None and source_files(root)!=job['source_hashes']:raise ValueError('source changed since packaging')
    return meta,job

def validate_submission(output,handle):
    if any(s in output.lower() for s in ('kernel push error','not valid dataset','invalid dataset')):raise ValueError('Kaggle rejected submission')
    version=re.search(r'Kernel version (\d+) successfully pushed',output)
    url=re.search(r'https://(?:www\.)?kaggle\.com/code/([A-Za-z0-9_-]+/[a-z0-9-]+)',output)
    if not version or int(version.group(1))<=0 or not url or url.group(1)!=handle:
        raise ValueError('positive kernel version and matching URL required')
    return dict(version=int(version.group(1)),url=url.group(0),acknowledged=True,running_verified=False)
