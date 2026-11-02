def test_runtime_identity_binds_labels_versions_and_frozen_analyzer():
    from reproduce import training as t
    assert hasattr(t,'runtime_identity'),'runtime data/analyzer identity binding is missing'
    rows=[dict(id='imagenet/a',task='ic',label=0,path='/mount/a.JPEG',split='train')]
    validation=[dict(id='imagenet/b',task='ic',label=1,path='/mount/b.JPEG',split='val')]
    inventory={'dataset_versions':{'owner/imagenet':1}}
    original=t.runtime_identity(rows,validation,inventory,'0'*64)
    assert original!=t.runtime_identity(rows,validation,inventory,'1'*64)
    changed=[dict(rows[0],label=2)]
    assert original!=t.runtime_identity(changed,validation,inventory,'0'*64)
    assert original!=t.runtime_identity(rows,validation,{'dataset_versions':{'owner/imagenet':2}},'0'*64)
    assert t.config_digest({'runtime_identity':original})!=t.config_digest({'runtime_identity':t.runtime_identity(changed,validation,inventory,'0'*64)})

def test_kaggle_payloads_pin_all_dataset_versions(tmp_path):
    from reproduce.kaggle import prepare
    from pathlib import Path
    import json
    root=Path(__file__).resolve().parents[1]
    for task in ('od','ic','ar','ot'):
        out=tmp_path/task;prepare(root,out,'baooo25r',f'paper-{task}-version-test','%s'%task,'pilot')
        refs=json.loads((out/'kernel-metadata.json').read_text())['dataset_sources']
        assert all(len(ref.split('/'))==3 and int(ref.split('/')[-1])>0 for ref in refs)
