"""Pinned source/bootstrap assets and exact dataset bindings for private Kaggle runs."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from .protocol import TRACKING_SOURCE

DATASETS={
 'od':['awsaf49/coco-2017-dataset'],
 'ic':['thbdh5765/ilsvrc2012','ashish2001/imagenet1kvalidation'],
 'ar':['qktttttttttt/kineticscleaned','jizeyong/kinetics400-rgb256-f10-val-data'],
 'ot':['abhimanyukarshni/got10k'],
}
DATASET_VERSIONS={'awsaf49/coco-2017-dataset':2,'thbdh5765/ilsvrc2012':1,
                  'ashish2001/imagenet1kvalidation':1,'qktttttttttt/kineticscleaned':4,
                  'jizeyong/kinetics400-rgb256-f10-val-data':1,'abhimanyukarshni/got10k':1}

def pinned_datasets(task):return [f'{ref}/{DATASET_VERSIONS[ref]}' for ref in DATASETS[task]]

PRROI_SOURCE='070a7950db6a945e30e8e3296204a1e975f131e8'
K400_VAL_URL='https://s3.amazonaws.com/kinetics/400/annotations/val.csv'
K400_VAL_SHA='358eaf47e7f80ebf9b17d49eb0635ad5e0fdab98a9cbd75ffdd2ee5d5e5b6944'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def safe_extract(archive,destination):
    destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=True)
    def check(name):
        p=(destination/name).resolve()
        if p!=destination and destination not in p.parents:raise ValueError('archive path outside destination')
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():check(info.filename)
            z.extractall(destination)
    else:
        with tarfile.open(archive) as t:
            for member in t.getmembers():
                check(member.name)
                if member.issym() or member.islnk():raise ValueError('archive links are not accepted')
            t.extractall(destination,filter='data')

def download(url,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not path.is_file():
        tmp=path.with_suffix(path.suffix+'.partial')
        request=urllib.request.Request(url,headers={'User-Agent':'paper-reproduction/0.1'})
        with urllib.request.urlopen(request,timeout=120) as response,tmp.open('wb') as f:
            shutil.copyfileobj(response,f)
        os.replace(tmp,path)
    return dict(url=url,path=str(path),bytes=path.stat().st_size,sha256=digest(path))

def command(args,log):
    with Path(log).open('a',encoding='utf-8') as f:
        result=subprocess.run([str(x) for x in args],stdout=f,stderr=subprocess.STDOUT,timeout=1200)
    if result.returncode:
        raise RuntimeError('asset setup failed: '+str(args[0])+'; '+Path(log).read_text(errors='replace')[-2500:])

def ensure_bpg(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if shutil.which('bpgenc') and shutil.which('bpgdec'):
        return dict(bpgenc=shutil.which('bpgenc'),bpgdec=shutil.which('bpgdec'),preinstalled=True)
    receipt=download('https://bellard.org/bpg/libbpg-0.9.8.tar.gz',directory/'libbpg-0.9.8.tar.gz')
    source=directory/'libbpg-0.9.8'
    if not source.is_dir():safe_extract(directory/'libbpg-0.9.8.tar.gz',directory)
    if not (source/'bpgenc').is_file() or not (source/'bpgdec').is_file():
        command(['make','-C',source,'-j2','USE_X265=','USE_JCTVC=y','bpgenc','bpgdec'],directory/'bpg_build.log')
    return dict(bpgenc=str(source/'bpgenc'),bpgdec=str(source/'bpgdec'),source=receipt)

def ensure_tracking(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    source=directory/f'pytracking-{TRACKING_SOURCE}'
    receipt=download(f'https://codeload.github.com/visionml/pytracking/zip/{TRACKING_SOURCE}',directory/'pytracking.zip')
    if not source.is_dir():safe_extract(directory/'pytracking.zip',directory)
    sub=source/'ltr/external/PreciseRoIPooling'
    sub_receipt=download(f'https://codeload.github.com/vacancy/PreciseRoIPooling/zip/{PRROI_SOURCE}',directory/'prroi.zip')
    if not (sub/'pytorch/prroi_pool/__init__.py').is_file():
        temp=directory/'prroi_source';safe_extract(directory/'prroi.zip',temp)
        extracted=temp/f'PreciseRoIPooling-{PRROI_SOURCE}'
        shutil.copytree(extracted,sub,dirs_exist_ok=True)
    # Source-only compatibility edits for current NumPy/PyTorch; record all changed files.
    patches=materialize_prroi_links(sub)
    for path in (source/'ltr').rglob('*.py'):
        raw=path.read_text();fixed=modernize_tracking_python(raw)
        if fixed!=raw:path.write_text(fixed);patches.append(str(path.relative_to(source)))
    for path in sub.rglob('*'):
        if path.suffix in ('.h','.c','.cpp','.cu','.cuh') and path.is_file():
            raw=path.read_text();fixed=modernize_prroi_source(raw)
            if fixed!=raw:path.write_text(fixed);patches.append(str(path.relative_to(source)))
    sys.path.insert(0,str(source))
    checkpoint=directory/'dimp50.pth'
    if not checkpoint.is_file():
        import gdown
        result=gdown.download(id='1qgachgqks2UGjKx-GdO1qylBDdB1f9KN',output=str(checkpoint),quiet=False)
        if not result:raise RuntimeError('official DiMP50 checkpoint download failed')
    if checkpoint.stat().st_size<1000000:raise ValueError('DiMP checkpoint is not a valid network asset')
    write_tracking_environment(source,directory)
    return dict(checkpoint=str(checkpoint),sha256=digest(checkpoint),source=receipt,
                prroi=sub_receipt,compatibility_patches=patches)

def write_tracking_environment(source,directory):
    source=Path(source);directory=Path(directory)
    # Legacy checkpoint unpickling imports ltr.admin.local even for inference.
    ltr=source/'ltr/admin/local.py';ltr.parent.mkdir(parents=True,exist_ok=True)
    ltr.write_text('class EnvironmentSettings:\n'
                   '    def __init__(self):\n'
                  f'        self.workspace_dir={str(directory)!r}\n'
                  f'        self.pretrained_networks={str(directory)!r}\n'
                  f'        self.tensorboard_dir={str(directory / "tensorboard")!r}\n')
    local=source/'pytracking/evaluation/local.py';local.parent.mkdir(parents=True,exist_ok=True)
    local.write_text('from pytracking.evaluation.environment import EnvSettings\n'
                     'def local_env_settings():\n'
                     '    settings=EnvSettings()\n'
                    f'    settings.network_path={str(directory)!r}\n'
                     '    return settings\n')

def modernize_prroi_source(source):
    # API-only migration: preserve the original precise pooling CUDA kernels.
    return (source.replace('#include <THC/THC.h>',
                           '#include <c10/cuda/CUDAException.h>\n#include <c10/cuda/CUDAStream.h>')
                  .replace('THCudaCheck(', 'C10_CUDA_CHECK(')
                  .replace('AT_CHECK(', 'TORCH_CHECK(')
                  .replace('.data<float>()','.data_ptr<float>()')
                  .replace('at::cuda::getCurrentCUDAStream','c10::cuda::getCurrentCUDAStream'))

def modernize_tracking_python(source):
    return source.replace('from torchvision.models.resnet import model_urls',
          'from torchvision.models import ResNet18_Weights, ResNet50_Weights, ResNet101_Weights\n'
          'model_urls={"resnet18":ResNet18_Weights.IMAGENET1K_V1.url, '
          '"resnet50":ResNet50_Weights.IMAGENET1K_V1.url, "resnet101":ResNet101_Weights.IMAGENET1K_V1.url}')

def materialize_prroi_links(root):
    # GitHub ZIP stores Git symlinks as small files containing relative targets.
    # Resolve only internal source links; otherwise nvcc tries compiling the path text.
    root=Path(root).resolve();changed=[]
    for path in root.rglob('*'):
        if not path.is_file() or path.suffix not in ('.h','.c','.cpp','.cu','.cuh'):continue
        raw=path.read_text().strip()
        if not raw.startswith('../') or '\n' in raw or len(raw)>250:continue
        target=(path.parent/raw).resolve()
        if root not in target.parents or not target.is_file():raise ValueError('invalid ROI source link')
        path.write_bytes(target.read_bytes());changed.append(str(path.relative_to(root)))
    return changed

def dataset_mount(inputs,reference):
    inputs=Path(inputs);owner,slug=reference.split('/')
    for path in (inputs/slug,inputs/'datasets'/owner/slug):
        if path.is_dir():return path
    raise FileNotFoundError('declared Kaggle dataset is not attached: '+reference)

def first_wnid_root(root):
    root=Path(root)
    for path in [root,root/'train',root/'val',root/'ILSVRC/Data/CLS-LOC/train']:
        if (path/'n01440764').is_dir():return path
    raise ValueError('canonical ImageNet class folders not found in '+str(root))

def records_for_task(task,inputs,asset_dir=None):
    from .data import coco_records,imagenet_records,kinetics_records,kinetics_frame_records,got_records,verify_splits
    mounts=[dataset_mount(inputs,ref) for ref in DATASETS[task]]
    if task=='ic':
        train=imagenet_records(first_wnid_root(mounts[0]),'train')
        validation=imagenet_records(first_wnid_root(mounts[1]),'val')
    elif task=='od':
        root=mounts[0];candidates=list(root.rglob('instances_train2017.json'))
        if len(candidates)!=1:raise ValueError('one original COCO train annotation file required')
        ann=candidates[0];base=ann.parent.parent
        images=lambda split:next((p for p in (base/split,root/split,base/'images'/split) if p.is_dir()),None)
        train=coco_records(images('train2017'),ann,'train')
        validation=coco_records(images('val2017'),ann.with_name('instances_val2017.json'),'val')
    elif task=='ar':
        annotation=Path(asset_dir or '/kaggle/working/paper_assets')/'k400_val.csv'
        label_receipt=download(K400_VAL_URL,annotation)
        if label_receipt['sha256']!=K400_VAL_SHA:raise ValueError('official K400 validation annotation SHA differs')
        train=kinetics_records(mounts[0],'train');validation=kinetics_frame_records(mounts[1],annotation)
        val_ids={r['id'] for r in validation};train=[r for r in train if r['id'] not in val_ids]
    else:
        root=mounts[0]
        if not (root/'train').is_dir():
            roots=[p.parent for p in root.rglob('train') if p.is_dir()]
            if len(roots)!=1:raise ValueError('one GOT10k train root required')
            root=roots[0]
        train=got_records(root,'train');validation=got_records(root,'val')
    verify_splits(train,validation)
    expected={'od':(118287,5000),'ic':(1281167,50000),'ot':(9335,180)}
    full=(task in expected and (len(train),len(validation))==expected[task])
    return train,validation,dict(datasets=DATASETS[task],mounts=[str(p) for p in mounts],
                                dataset_versions={ref:DATASET_VERSIONS[ref] for ref in DATASETS[task]},
                                train_sources=len(train),validation_sources=len(validation),
                                full_original_training_corpus=full,
                                scope_note='K400 TRAIN subset; validation JPEG mirror joined to official labels; provider cadence unverified' if task=='ar' else 'original dataset family; file counts recorded',
                                **({'validation_annotations':label_receipt,'validation_source_format':'provider_jpeg_frames',
                                    'validation_provider_cadence':'unverified'} if task=='ar' else {}))
