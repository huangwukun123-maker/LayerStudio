"""Local 512-pixel LaMa inference with bounded, mask-only replacement.

Input is uint8 RGB. The Apache-2.0 OpenCV LaMa ONNX model accepts normalized
float RGB NCHW plus a binary float mask and returns RGB NCHW in [0,255].
Estimated hidden pixels are not recovered source artwork. This module never
downloads models or modifies visible pixels outside the caller's erase mask.
"""
from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime as ort


class LocalInpainter:
    def __init__(self, model_path, *, threads=4):
        self.model_path=Path(model_path)
        self.threads=max(1,min(4,int(threads)))
        self._session=None
        self.last_info=None

    def _load(self):
        if self._session is None:
            options=ort.SessionOptions()
            options.intra_op_num_threads=self.threads
            options.inter_op_num_threads=1
            options.log_severity_level=3
            self._session=ort.InferenceSession(str(self.model_path),sess_options=options,
                                               providers=['CPUExecutionProvider'])
        return self._session

    def inpaint(self,rgb,mask):
        rgb=np.asarray(rgb)
        mask=np.asarray(mask)
        if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or min(rgb.shape[:2])<1:
            raise ValueError('Inpainting requires a non-empty uint8 RGB image')
        if mask.ndim!=2 or mask.shape!=rgb.shape[:2] or not np.isfinite(mask).all():
            raise ValueError('Inpainting mask must match image height and width')
        active=mask>0
        if not np.any(active):
            self.last_info={'passes':0,'seconds':0.,'provider':'CPUExecutionProvider'}
            return rgb.copy()
        begin=time.perf_counter()
        height,width=rgb.shape[:2]
        scale=512/max(height,width)
        new_width=max(1,round(width*scale));new_height=max(1,round(height*scale))
        resized=cv2.resize(rgb,(new_width,new_height),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_CUBIC)
        # Area sampling of float masks keeps isolated masked pixels when
        # downscaling, whereas nearest-neighbour can silently lose them.
        resized_mask=(cv2.resize(active.astype(np.float32),(new_width,new_height),
                                  interpolation=cv2.INTER_AREA)>0).astype(np.float32)
        bottom=512-new_height;right=512-new_width
        padded=cv2.copyMakeBorder(resized,0,bottom,0,right,cv2.BORDER_REFLECT_101)
        # Reflect BOTH arrays. A zero-filled padded mask would expose reflected
        # copies of erased letters and cause the model to copy those letters back.
        padded_mask=cv2.copyMakeBorder(resized_mask,0,bottom,0,right,cv2.BORDER_REFLECT_101)
        feed={'image':np.ascontiguousarray(padded.transpose(2,0,1)[None],dtype=np.float32)/255,
              'mask':np.ascontiguousarray(padded_mask[None,None],dtype=np.float32)}
        output=np.asarray(self._load().run(None,feed)[0])
        if output.shape!=(1,3,512,512) or not np.isfinite(output).all():
            raise RuntimeError('Background model returned an invalid image')
        estimate=np.clip(output[0].transpose(1,2,0)[:new_height,:new_width],0,255)
        estimate=cv2.resize(estimate,(width,height),interpolation=cv2.INTER_CUBIC)
        result=rgb.copy()
        result[active]=np.clip(estimate[active],0,255).round().astype(np.uint8)
        self.last_info={'passes':1,'seconds':round(time.perf_counter()-begin,3),
                        'provider':'CPUExecutionProvider'}
        return result
