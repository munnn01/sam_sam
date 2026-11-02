"""Measured BPG and low-delay x264/x265 streams; never report surrogate bpp as actual."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import os
import subprocess
import tempfile
import numpy as np
from PIL import Image
import torch
from .protocol import QPS

@dataclass
class CodecOutput:
    reconstruction:torch.Tensor
    bpp:float
    bytes:int
    stream:str
    sha256:str

def run(args,raw=None):
    result=subprocess.run([str(a) for a in args],input=raw,capture_output=True,timeout=180)
    if result.returncode:
        raise RuntimeError('codec failed: '+result.stderr.decode(errors='replace')[-2500:])
    return result.stdout

class RealCodec:
    def __init__(self,ffmpeg='ffmpeg',bpgenc='bpgenc',bpgdec='bpgdec',preset='medium',fps=25,gop=16):
        self.ffmpeg=ffmpeg;self.bpgenc=bpgenc;self.bpgdec=bpgdec
        self.preset=preset;self.fps=fps;self.gop=gop

    def __call__(self,x,codec,qp,directory=None):
        if x.ndim!=5 or x.shape[0]!=1 or x.shape[2]!=3 or qp not in QPS or codec not in ('bpg','h264','h265'):
            raise ValueError('single N,T,RGB,H,W sample and configured real codec/QP required')
        if codec=='bpg' and x.shape[1]!=1:raise ValueError('BPG is an image codec, not a video stream')
        if directory is None:
            with tempfile.TemporaryDirectory(prefix='paper-codec-',dir=os.environ.get('PAPER_CODEC_TMP')) as temp:
                result=self(x,codec,qp,Path(temp))
                result.stream=''  # ephemeral stream path must not be presented as a retained artifact
                return result
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        array=(x.detach().cpu().clamp(0,1)[0].permute(0,2,3,1).numpy()*255).round().astype(np.uint8)
        t,h,w,_=array.shape
        if codec=='bpg':
            source=directory/'source.png';stream=directory/'stream.bpg';decoded=directory/'decoded.png'
            Image.fromarray(array[0]).save(source)
            run([self.bpgenc,'-q',qp,'-o',stream,source]);run([self.bpgdec,'-o',decoded,stream])
            with Image.open(decoded) as image:result=np.asarray(image.convert('RGB'))[None].copy()
            # Keep actual compressed streams for auditing; avoid hundreds of thousands of duplicate PNGs.
            source.unlink();decoded.unlink()
        else:
            ph=h+h%2;pw=w+w%2
            padded=np.pad(array,((0,0),(0,ph-h),(0,pw-w),(0,0)),mode='edge')
            stream=directory/('stream.h264' if codec=='h264' else 'stream.hevc')
            options=[self.ffmpeg,'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24',
                     '-s',f'{pw}x{ph}','-r',self.fps,'-i','pipe:0','-an','-c:v','libx264' if codec=='h264' else 'libx265',
                     '-preset',self.preset,'-qp',qp,'-pix_fmt','yuv420p','-g',self.gop,'-bf','0','-threads','2']
            key='-x264-params' if codec=='h264' else '-x265-params'
            params=f'scenecut=0:keyint={self.gop}:min-keyint={self.gop}:bframes=0'
            if codec=='h265':params+=':log-level=error:pools=1:frame-threads=1'
            run(options+[key,params,stream],padded.tobytes())
            raw=run([self.ffmpeg,'-hide_banner','-loglevel','error','-i',stream,'-f','rawvideo','-pix_fmt','rgb24','-threads','2','pipe:1'])
            if len(raw)!=t*ph*pw*3:raise ValueError('decoded real stream frame count differs from input')
            result=np.frombuffer(raw,dtype=np.uint8).reshape(t,ph,pw,3)[:,:h,:w].copy()
        if result.shape!=(t,h,w,3):raise ValueError('codec changed input geometry')
        content=stream.read_bytes();count=len(content)
        reconstruction=torch.from_numpy(result).permute(0,3,1,2).float()[None].to(x.device)/255
        return CodecOutput(reconstruction,count*8/(t*h*w),count,str(stream),hashlib.sha256(content).hexdigest())
