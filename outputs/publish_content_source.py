"""Hash-preserving source publication manifest; all artifacts remain in SAM."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'outputs/content_rate_20261101';BASE.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT))
from reproduce.kaggle import source_files
if sys.argv[1]=='prepare':
    hashes=source_files(ROOT)
    baseline=json.loads(Path('D:/STUDY/LAB/bao_1/output/paper_reproduction_20261101/seg_track_sam/codecs/jobs/is-h264/payload/job.json').read_text())['source_hashes']
    original=json.loads(Path('D:/STUDY/LAB/bao_1/output/paper_reproduction_20261028/legacy_source_manifest.json').read_text())['source_hashes']
    if not all(hashes[n]==v for n,v in original.items()):raise ValueError('original core sources changed')
    source=hashlib.sha256(json.dumps(hashes,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    (ROOT/'SOURCE_MANIFEST.json').write_text(json.dumps(dict(source_sha256=source,source_hashes=hashes),indent=2))
    changed=[n for n,v in hashes.items() if baseline.get(n)!=v]+['SOURCE_MANIFEST.json']
    elements=[];proof=[]
    for name in changed:
        raw=(ROOT/name).read_bytes()
        elements.append(dict(path='paper_reproduction/'+name,mode='100644',type='blob',content=raw.decode('utf-8')))
        proof.append(dict(path='paper_reproduction/'+name,sha=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()))
    (BASE/'publish_tree.json').write_text(json.dumps(elements,ensure_ascii=True),encoding='utf-8')
    (BASE/'publish_proof.json').write_text(json.dumps(dict(source_sha256=source,source_files=len(hashes),original_core_unchanged=len(original),files=proof),indent=2))
    print(json.dumps(dict(source_sha256=source,source_files=len(hashes),changed=changed)))
else:print(json.dumps(json.loads((BASE/'publish_tree.json').read_text(encoding='utf-8')),ensure_ascii=True))
