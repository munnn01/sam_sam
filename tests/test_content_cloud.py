import json
from pathlib import Path
import pytest
from reproduce.coco_content_rate_study import config
from reproduce.coco_content_rate_cloud import prepare,verify

def test_content_payload_binds_codec_calibration_and_runner(tmp_path):
    root=Path(__file__).resolve().parents[1]
    for task in ('is','od'):
        for codec in ('h264','h265'):
            path=tmp_path/(task+codec);cfg=config(task,'control',codec)
            prepare(root,path,'testowner','content-codec',cfg)
            _,job=verify(path,root)
            assert job['config']['rate_profile']=='nal_content_v1'
            code=''.join(json.loads((path/'notebook.ipynb').read_text())['cells'][0]['source'])
            assert "'reproduce.coco_content_rate_study'" in code
            job['config']['rate_median_error_max']=10
            (path/'job.json').write_text(json.dumps(job))
            with pytest.raises(ValueError):verify(path)
