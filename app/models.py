"""Offline inference models. All paths resolved against the application bundle."""
import os
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
import cv2
from PIL import Image

APP_DIR = Path(sys.executable).resolve().parent if getattr(sys,'frozen',False) else Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / 'models'
NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)


def model_status():
    return [
        {'name': '中文文字识别', 'ready': all((MODEL_DIR/'ocr'/n).is_file() for n in
          ['PP-OCRv6_det_small.onnx', 'PP-OCRv6_rec_small.onnx', 'ch_ppocr_mobile_v2.0_cls_mobile.onnx'])},
        {'name': '主体分割', 'ready': (MODEL_DIR/'u2net.onnx').is_file()},
        {'name': '文字底图修复', 'ready': (MODEL_DIR/'inpaint/inpainting_lama_2025jan.onnx').is_file()},
        {'name': 'AI清晰化', 'ready': all((MODEL_DIR/'esrgan'/p).is_file() for p in
          ['realesrgan-ncnn-vulkan.exe', 'models/realesrgan-x4plus.bin', 'models/realesrgan-x4plus.param'])},
    ]


class LocalModels:
    def __init__(self):
        self.ocr = None
        self.segmenter = None
        self.inpainter = None

    def inpaint(self, rgb, mask):
        if self.inpainter is None:
            from inpaint_model import LocalInpainter
            self.inpainter=LocalInpainter(MODEL_DIR/'inpaint/inpainting_lama_2025jan.onnx')
        return self.inpainter.inpaint(rgb,mask)

    def recognize(self, rgb):
        if self.ocr is None:
            from rapidocr import RapidOCR
            paths = MODEL_DIR / 'ocr'
            self.ocr = RapidOCR(params={
                'Det.model_path': str(paths/'PP-OCRv6_det_small.onnx'),
                'Rec.model_path': str(paths/'PP-OCRv6_rec_small.onnx'),
                'Cls.model_path': str(paths/'ch_ppocr_mobile_v2.0_cls_mobile.onnx'),
                'EngineConfig.onnxruntime.intra_op_num_threads': 4,
                'EngineConfig.onnxruntime.inter_op_num_threads': 1,
                'Global.log_level': 'warning',
                'Det.unclip_ratio': 1.1,
            })
        # RapidOCR ndarray input is BGR.
        result = self.ocr(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if result.boxes is None or result.txts is None:
            return []
        return [{'quad': box.tolist(), 'text': str(text), 'score': float(score)}
                for box, text, score in zip(result.boxes, result.txts, result.scores)]

    def foreground(self, rgb):
        if self.segmenter is None:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 4
            opts.inter_op_num_threads = 1
            self.segmenter = ort.InferenceSession(str(MODEL_DIR/'u2net.onnx'), opts,
                                                   providers=['CPUExecutionProvider'])
        small = np.asarray(Image.fromarray(rgb).resize((320, 320), Image.Resampling.LANCZOS), dtype=np.float32)
        small /= max(float(small.max()), 1)
        small = (small - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
        tensor = small.transpose(2, 0, 1)[None].astype(np.float32)
        pred = self.segmenter.run(None, {self.segmenter.get_inputs()[0].name: tensor})[0][0, 0]
        pred = (pred - pred.min()) / max(float(pred.max()-pred.min()), 1e-6)
        alpha = cv2.resize(pred, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
        # Crisp interiors keep background reconstruction from changing solid objects.
        alpha = np.clip((alpha-.08)/.84, 0, 1)
        return np.round(alpha*255).astype(np.uint8)

    def enhance(self, input_path, output_path, progress):
        root = MODEL_DIR / 'esrgan'
        # Auto selection normally picks the discrete GPU; this computer's Vulkan
        # enumeration has NVIDIA at index 1. Environment is optional override.
        gpu = os.environ.get('LAYERSTUDIO_GPU', '')
        command = [str(root/'realesrgan-ncnn-vulkan.exe'), '-i', str(Path(input_path).resolve()),
                   '-o', str(Path(output_path).resolve()), '-m', str(root/'models'),
                   '-n', 'realesrgan-x4plus', '-s', '4', '-t', '256']
        if gpu:
            command += ['-g', gpu]
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding='utf-8', errors='replace', creationflags=NO_WINDOW)
        log = []
        import threading
        timed_out = threading.Event()
        def timeout():
            timed_out.set()
            proc.kill()
        timer = threading.Timer(300, timeout)
        timer.start()
        try:
            for line in proc.stdout:
                log.append(line.rstrip())
                match = re.match(r'^\s*(\d+(?:\.\d+)?)%\s*$', line)
                if match:
                    progress(float(match.group(1)))
            code = proc.wait()
        finally:
            timer.cancel()
            if proc.stdout: proc.stdout.close()
        Path(output_path).with_suffix('.log').write_text('\n'.join(log), encoding='utf-8')
        if timed_out.is_set() or code or not Path(output_path).is_file():
            raise RuntimeError('本机清晰化失败，请确认显卡驱动支持Vulkan。任务未生成冒充AI修复的文件。')
        with Image.open(output_path) as image:
            arr = np.asarray(image.convert('RGB'))
            if float(arr.std()) < .1:
                raise RuntimeError('清晰化模型返回空白图片，请更新显卡驱动后重试。')
        return 'Real-ESRGAN 本机AI超分辨率（细节为估算增强）'
