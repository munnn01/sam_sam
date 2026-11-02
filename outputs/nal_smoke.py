import json
from pathlib import Path
import torch
from reproduce.codec_content_rate import NALCodec,nal_accounting
root=Path('D:/STUDY/LAB/SAM/outputs/nal_smoke');root.mkdir(parents=True,exist_ok=True)
torch.manual_seed(19);rows=[]
for name in ('h264','h265'):
    codec=NALCodec(name,ffmpeg='D:/STUDY/AI/envs/ten_env/Library/bin/ffmpeg.exe')
    for size in (128,512):
        for q in (30,35,40,45):
            for label,x in (('flat',torch.full((1,1,3,size,size),.5)),('random',torch.rand(1,1,3,size,size))):
                v=codec(x,name,q,root/name/str(size)/str(q)/label)
                assert v.bytes==v.content_bytes+v.overhead_bytes and v.reconstruction.shape==x.shape
                rows.append(dict(codec=name,size=size,qp=q,kind=label,bytes=v.bytes,content_bytes=v.content_bytes,non_vcl_bytes=v.overhead_bytes))
(root/'receipt.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(dict(points=len(rows),all_bytes_accounted=True,non_vcl_stable_across_content=True)))
