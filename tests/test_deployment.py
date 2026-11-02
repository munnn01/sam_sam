import importlib.util
import json
from pathlib import Path
import pytest

def api():
    assert importlib.util.find_spec('reproduce.kaggle') is not None,'Kaggle launcher not implemented'
    return __import__('reproduce.kaggle',fromlist=['*'])

def test_private_payload_binds_original_task_data_and_four_qps(tmp_path):
    k=api();root=Path(__file__).resolve().parents[1]
    job=k.prepare(root,tmp_path/'payload','baooo25r','paper-video-ar-qp4-20261023','ar','pilot')
    meta,verified=k.verify_payload(tmp_path/'payload',root)
    assert meta['is_private'] is True and meta['enable_gpu'] is True
    assert meta['dataset_sources']==['qktttttttttt/kineticscleaned/4','jizeyong/kinetics400-rgb256-f10-val-data/1']
    assert verified['config']['qps']==[30,35,40,45] and verified['config']['task']=='ar'
    assert verified['source_sha256']==job['source_sha256']

def test_modified_notebook_or_metadata_is_rejected(tmp_path):
    k=api();root=Path(__file__).resolve().parents[1]
    k.prepare(root,tmp_path/'payload','baooo25r','paper-image-od-qp4-20261023','od','pilot')
    p=tmp_path/'payload/kernel-metadata.json';m=json.loads(p.read_text());m['is_private']=False;p.write_text(json.dumps(m))
    with pytest.raises(ValueError):k.verify_payload(tmp_path/'payload',root)

def test_submission_requires_positive_version_and_matching_handle():
    k=api();handle='baooo25r/paper-image-od-qp4-20261023'
    good=f'Kernel version1 successfully pushed. Please check progress at https://www.kaggle.com/code/{handle}'
    # Exact Kaggle CLI success wording is part of the boundary.
    good=good.replace('version1','version 1')
    assert k.validate_submission(good,handle)['version']==1
    with pytest.raises(ValueError):k.validate_submission('Kernel push error: GPU quota',handle)
    with pytest.raises(ValueError):k.validate_submission(good,handle+'-wrong')
