from types import SimpleNamespace
import torch
from reproduce.codec_content_rate import ContentProxy
from reproduce.coco_content_rate_study import calibrate_content,audit_content,config
class Bank(torch.nn.Module):
    def __init__(self):
        super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(()))
    def forward(self,x,q):return SimpleNamespace(reconstruction=x*.95,bpp=x.mean()+.1+self.anchor*0)
class Codec:
    def overhead(self,x,q):return 100
    def __call__(self,x,c,q):
        pixels=x.shape[-2]*x.shape[-1]
        content=round(float(2*(x.mean()+.1))*pixels/8)
        return SimpleNamespace(content_bytes=content,overhead_bytes=100,bytes=content+100,
            bpp=(content+100)*8/pixels,reconstruction=x*.95)
class NPP(torch.nn.Module):
    def forward(self,x,q):return x*.8
class Budget:
    def require(self,stage):pass

def test_full_geometry_fit_uses_train_only_and_audit_is_paired(tmp_path):
    codec=Codec();proxy=ContentProxy(Bank(),codec);npp=NPP();cfg=config('od','control','h265')
    cfg['rate_fit_sources']=2
    train=[(torch.full((1,1,3,16,32),v),{},dict(id='train/'+str(i))) for i,v in enumerate((.2,.5))]
    dev=[(torch.full((1,1,3,16,32),v),{},dict(id='dev/'+str(i))) for i,v in enumerate((.3,.6))]
    cpu_rng=torch.get_rng_state().clone()
    fitted=calibrate_content(proxy,npp,train,cfg,codec,tmp_path,Budget())
    assert torch.equal(cpu_rng,torch.get_rng_state())
    assert fitted['rows']==32 and all(abs(s-2)<.05 for s in fitted['scales'])
    result=audit_content(proxy,npp,dev,cfg,codec,tmp_path,Budget(),'initial')
    assert len(result['rows'])==16 and len(result['paired_changes'])==8
    assert all(r['source'].startswith('dev/') for r in result['rows'])
    assert result['gate']['passed'] and result['direction']['aligned']==16
    assert npp.training
