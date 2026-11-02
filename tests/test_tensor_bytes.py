import hashlib,struct
import pytest
import torch

def test_tensor_and_model_hashes_preserve_raw_bytes_without_storage_iteration(monkeypatch):
    from reproduce.sam2_masks import tensor_digest
    from reproduce.proxy_recovery import model_digest
    x=torch.tensor([[1.,2.],[3.,4.]])
    raw=struct.pack('=4f',1,2,3,4)
    expected=hashlib.sha256(str((x.dtype,tuple(x.shape))).encode()+raw).hexdigest()
    model=torch.nn.Linear(2,2,bias=False)
    with torch.no_grad():model.weight.copy_(x)
    expected_model=hashlib.sha256(b'weight'+str((2,2)).encode()+raw).hexdigest()
    monkeypatch.setattr(torch.Tensor,'untyped_storage',lambda self:pytest.fail('per-byte storage hashing forbidden'))
    assert tensor_digest(x)==expected
    assert tensor_digest(x.clone())==expected
    assert model_digest(model)==expected_model

def test_raw_bytes_handle_offset_noncontiguous_and_empty_tensors():
    import importlib.util
    assert importlib.util.find_spec('reproduce.tensor_bytes') is not None,'fast raw tensor byte reader missing'
    from reproduce.tensor_bytes import raw_tensor_bytes
    values=torch.tensor([[1,2,3],[4,5,6]],dtype=torch.int16)
    assert raw_tensor_bytes(values[:,1:])==struct.pack('=4h',2,3,5,6)
    assert raw_tensor_bytes(values.flatten()[2:5])==struct.pack('=3h',3,4,5)
    assert raw_tensor_bytes(torch.empty(0))==b''
