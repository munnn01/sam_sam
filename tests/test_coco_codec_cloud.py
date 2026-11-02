from pathlib import Path
import json
import pytest
from reproduce.coco_codec_study import config
from reproduce.coco_codec_cloud import prepare,verify

def test_paired_payload_binding_and_tamper(tmp_path):
 root=Path(__file__).resolve().parents[1]
 for task in ('is','od'):
  for codec in ('h264','h265'):
   folder=tmp_path/(task+codec)
   prepare(root,folder,'testowner','codec-study',config(task,'control',codec))
   _,j=verify(folder,root)
   assert j['config']['codecs']==[codec] and j['input_spec']['visibility']=='public'
   notebook=json.loads((folder/'notebook.ipynb').read_text());code=''.join(notebook['cells'][0]['source'])
   assert "'reproduce.coco_codec_study'" in code and "'reproduce.tri_task_study'" not in code
   j['config']['codecs']=['bpg'];(folder/'job.json').write_text(json.dumps(j))
   with pytest.raises(ValueError):verify(folder)
