"""Private image audit payload with a separately verified frozen continuation source."""
import ast
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import zlib

from .kaggle import prepare, verify_payload
from .proxy_cloud import OLD_COMMAND


IMAGE_COMMAND="""    owner,source_slug,source_version=SOURCE_KERNEL.split('/')
    roots=[pathlib.Path('/kaggle/input')/source_slug,
           pathlib.Path('/kaggle/input/notebooks')/owner/source_slug,
           pathlib.Path('/kaggle/input/kernels')/owner/source_slug]
    candidates=set()
    for mount in roots:
        for base in (mount,mount/source_version):
            for relative in ('paper_results/last.pt','output/paper_results/last.pt','last.pt'):
                path=base/relative
                if path.is_file():candidates.add(path.resolve())
    def file_sha(path):
        h=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
        return h.hexdigest()
    matches=[p for p in sorted(candidates) if file_sha(p)==CHECKPOINT_SHA]
    if len(matches)!=1:raise ValueError('owned immutable image checkpoint missing or ambiguous')
    legacy_contents=json.loads(zlib.decompress(base64.b64decode(LEGACY_BUNDLE)))
    if set(legacy_contents)!=set(LEGACY_MANIFEST):raise ValueError('legacy file inventory differs')
    if hashlib.sha256(json.dumps(LEGACY_MANIFEST,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=LEGACY_SHA:
        raise ValueError('legacy manifest differs')
    legacy_root=pathlib.Path('/kaggle/working/legacy_image_reproduction')
    for name,content in legacy_contents.items():
        path=legacy_root/name
        if legacy_root.resolve() not in path.resolve().parents:raise ValueError('unsafe legacy path')
        if hashlib.sha256(content.encode()).hexdigest()!=LEGACY_MANIFEST[name]:raise ValueError('legacy source differs')
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content,encoding='utf-8')
    print(json.dumps({'stage':'legacy_image_verified','task':TASK,'source_sha256':LEGACY_SHA,
                      'checkpoint_sha256':CHECKPOINT_SHA}),flush=True)
    args=[sys.executable,'-u','-m','reproduce.image_proxy_audit','--task',TASK,
          '--checkpoint',str(matches[0]),'--checkpoint-sha256',CHECKPOINT_SHA,
          '--training-source-sha256',LEGACY_SHA,'--legacy-root',str(legacy_root),'--continue-proxy',
          '--max-seconds',str(IMAGE_OPTIONS['max_seconds']),
          '--audit-seconds',str(IMAGE_OPTIONS['audit_seconds']),'--out',str(OUT)]
"""


def validate_legacy(bundle,hashes,sha):
    if not re.fullmatch(r'[0-9a-f]{64}',sha):raise ValueError('exact legacy SHA required')
    actual=hashlib.sha256(json.dumps(hashes,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if actual!=sha:raise ValueError('frozen legacy manifest SHA differs')
    contents=json.loads(zlib.decompress(base64.b64decode(bundle,validate=True)))
    if set(contents)!=set(hashes) or 'reproduce/run.py' not in contents:raise ValueError('legacy inventory differs')
    for name,content in contents.items():
        path=PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
            raise ValueError('unsafe legacy source path')
        if hashlib.sha256(content.encode()).hexdigest()!=hashes[name]:raise ValueError('legacy file SHA differs')


def prepare_image_diagnostic(root,directory,account,slug,task,source_kernel,checkpoint_sha,
                             training_source,legacy_bundle,legacy_hashes,max_seconds=18000,audit_seconds=5400):
    if task not in ('od','ic') or not 0<audit_seconds<=5400 or not audit_seconds<=max_seconds<=18000:
        raise ValueError('invalid image task/budget')
    if not re.fullmatch(re.escape(account)+r'/[a-z0-9-]+/[1-9][0-9]*',source_kernel):
        raise ValueError('owned version-pinned kernel source required')
    if not re.fullmatch(r'[0-9a-f]{64}',checkpoint_sha):raise ValueError('exact checkpoint SHA required')
    validate_legacy(legacy_bundle,legacy_hashes,training_source)
    directory=Path(directory);job=prepare(root,directory,account,slug,task,'full')
    settings=dict(max_seconds=max_seconds,audit_seconds=audit_seconds,training_source_sha256=training_source)
    constants=dict(SOURCE_KERNEL=source_kernel,CHECKPOINT_SHA=checkpoint_sha,LEGACY_SHA=training_source,
        LEGACY_MANIFEST=legacy_hashes,LEGACY_BUNDLE=legacy_bundle,IMAGE_OPTIONS=settings)
    notebook=json.loads((directory/'notebook.ipynb').read_text());code=''.join(notebook['cells'][0]['source'])
    if code.count(OLD_COMMAND)!=1:raise ValueError('canonical bootstrap command changed')
    code=''.join(f'{name}={value!r}\n' for name,value in constants.items())+code.replace(OLD_COMMAND,IMAGE_COMMAND)
    compile(code,'image-audit-bootstrap','exec');notebook['cells'][0]['source']=code.splitlines(True)
    meta=json.loads((directory/'kernel-metadata.json').read_text());meta['kernel_sources']=[source_kernel]
    job.update(image_diagnostic=settings,checkpoint_source=source_kernel,checkpoint_sha256=checkpoint_sha)
    for name,value in [('notebook.ipynb',notebook),('kernel-metadata.json',meta)]:
        (directory/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    job['payload_hashes']={name:hashlib.sha256((directory/name).read_bytes()).hexdigest()
        for name in ('notebook.ipynb','kernel-metadata.json')}
    (directory/'job.json').write_text(json.dumps(job,indent=2))
    verify_image_diagnostic(directory,root);return job


def verify_image_diagnostic(directory,root=None):
    directory=Path(directory);meta,job=verify_payload(directory,root)
    source=''.join(json.loads((directory/'notebook.ipynb').read_text())['cells'][0]['source']);constants={}
    names={'SOURCE_KERNEL','CHECKPOINT_SHA','LEGACY_SHA','LEGACY_MANIFEST','LEGACY_BUNDLE','IMAGE_OPTIONS',
           'SOURCE_SHA','MANIFEST','BUNDLE'}
    for node in ast.parse(source).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in names:
            constants[node.targets[0].id]=ast.literal_eval(node.value)
    if (job['task'] not in ('od','ic') or constants['IMAGE_OPTIONS']!=job['image_diagnostic']
        or constants['SOURCE_KERNEL']!=job['checkpoint_source'] or meta['kernel_sources']!=[job['checkpoint_source']]
        or not re.fullmatch(re.escape(meta['id'].split('/')[0])+r'/[a-z0-9-]+/[1-9][0-9]*',job['checkpoint_source'])
        or constants['CHECKPOINT_SHA']!=job['checkpoint_sha256']
        or constants['LEGACY_SHA']!=job['image_diagnostic']['training_source_sha256']
        or constants['SOURCE_SHA']!=job['source_sha256'] or constants['MANIFEST']!=job['source_hashes']
        or OLD_COMMAND in source or IMAGE_COMMAND not in source):raise ValueError('image source/command binding differs')
    validate_legacy(constants['LEGACY_BUNDLE'],constants['LEGACY_MANIFEST'],constants['LEGACY_SHA'])
    validate_legacy(constants['BUNDLE'],constants['MANIFEST'],constants['SOURCE_SHA'])
    compile(source,'verified-image-bootstrap','exec');return meta,job
