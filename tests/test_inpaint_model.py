import sys
from pathlib import Path
import cv2
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))


def fake_runtime(monkeypatch):
    import onnxruntime
    calls=[]
    class Session:
        def __init__(self,*args,**kwargs): calls.append({'load':kwargs})
        def run(self,names,feeds):
            calls.append({'feeds':{name:value.copy() for name,value in feeds.items()}})
            output=np.empty((1,3,512,512),np.float32)
            output[:,0]=65;output[:,1]=85;output[:,2]=105
            return [output]
    monkeypatch.setattr(onnxruntime,'InferenceSession',Session)
    return calls


def test_contract_padding_does_not_expose_reflected_text_and_preserves_unmasked_pixels(monkeypatch,tmp_path):
    from inpaint_model import LocalInpainter
    calls=fake_runtime(monkeypatch)
    rgb=np.full((60,200,3),[12,40,240],np.uint8)
    mask=np.zeros((60,200),np.uint8);mask[20:50,40:180]=255
    original=rgb.copy();initialmask=mask.copy()
    runner=LocalInpainter(tmp_path/'fake.onnx')
    result=runner.inpaint(rgb,mask)
    assert np.array_equal(rgb,original) and np.array_equal(mask,initialmask)
    assert np.array_equal(result[mask==0],rgb[mask==0])
    assert np.all(result[mask>0]==[65,85,105])
    feed=calls[1]['feeds']
    assert feed['image'].shape==(1,3,512,512)
    assert feed['mask'].shape==(1,1,512,512)
    assert feed['image'].dtype==np.float32 and feed['mask'].dtype==np.float32
    assert np.allclose(feed['image'][0,:,0,0],np.array([12,40,240])/255)
    resized=cv2.resize(mask.astype(np.float32),(512,154),interpolation=cv2.INTER_AREA)>0
    expected=cv2.copyMakeBorder(resized.astype(np.float32),0,358,0,0,cv2.BORDER_REFLECT_101)
    assert np.array_equal(feed['mask'][0,0],expected), 'reflected image and mask must agree or erased text leaks into padding'
    runner.inpaint(rgb,mask)
    assert sum('load' in call for call in calls)==1,'model session is reused across images'


def test_empty_mask_never_loads_model(monkeypatch,tmp_path):
    from inpaint_model import LocalInpainter
    calls=fake_runtime(monkeypatch)
    rgb=np.full((27,35,3),143,np.uint8)
    result=LocalInpainter(tmp_path/'missing.onnx').inpaint(rgb,np.zeros((27,35),np.uint8))
    assert np.array_equal(result,rgb) and result is not rgb
    assert calls==[]


def test_tiny_mask_survives_downsampling(monkeypatch,tmp_path):
    from inpaint_model import LocalInpainter
    calls=fake_runtime(monkeypatch)
    rgb=np.full((1024,1024,3),143,np.uint8)
    mask=np.zeros((1024,1024),np.uint8);mask[100,100]=255
    result=LocalInpainter(tmp_path/'fake.onnx').inpaint(rgb,mask)
    assert calls[1]['feeds']['mask'].sum()>=1
    assert np.array_equal(result[mask==0],rgb[mask==0])


def test_invalid_shape_or_dtype_is_rejected_before_inference(monkeypatch,tmp_path):
    from inpaint_model import LocalInpainter
    calls=fake_runtime(monkeypatch)
    runner=LocalInpainter(tmp_path/'fake.onnx')
    with pytest.raises(ValueError):runner.inpaint(np.zeros((10,12,4),np.uint8),np.zeros((10,12),np.uint8))
    with pytest.raises(ValueError):runner.inpaint(np.zeros((10,12,3),np.float32),np.zeros((10,12),np.uint8))
    with pytest.raises(ValueError):runner.inpaint(np.zeros((10,12,3),np.uint8),np.zeros((12,10),np.uint8))
    assert calls==[]


def test_invalid_model_output_raises_without_mutating_source(monkeypatch,tmp_path):
    from inpaint_model import LocalInpainter
    import onnxruntime
    class BadSession:
        def __init__(self,*args,**kwargs): pass
        def run(self,*args): return [np.full((1,3,512,512),np.nan,np.float32)]
    monkeypatch.setattr(onnxruntime,'InferenceSession',BadSession)
    rgb=np.full((20,20,3),143,np.uint8)
    before=rgb.copy()
    with pytest.raises(RuntimeError,match='invalid image'):
        LocalInpainter(tmp_path/'bad.onnx').inpaint(rgb,np.ones((20,20),np.uint8))
    assert np.array_equal(rgb,before)
