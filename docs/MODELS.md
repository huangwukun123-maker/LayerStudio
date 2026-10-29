# 模型和运行文件准备

应用仅从 `app/models` 读取模型，不在启动时自动下载或执行安装脚本。该目录已被 Git 忽略；请不要把现有软件的整个 data、runtime 或授权配置复制进仓库。

## 预期目录

```text
app/models/
  ocr/
    PP-OCRv6_det_small.onnx
    PP-OCRv6_rec_small.onnx
    ch_ppocr_mobile_v2.0_cls_mobile.onnx
  u2net.onnx
  inpaint/
    inpainting_lama_2025jan.onnx
  esrgan/
    realesrgan-ncnn-vulkan.exe
    models/
      realesrgan-x4plus.bin
      realesrgan-x4plus.param
```

Real-ESRGAN 上游发行包的必要 DLL 应按其说明与程序放在同一目录，不要从不明 DLL 下载站获取。

## 来源与边界

| 资源 | 来源 | 当前说明 |
|---|---|---|
| OCR | [RapidOCR](https://github.com/RapidAI/RapidOCR)、[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | 使用兼容当前 RapidOCR 配置的 ONNX 文件；不能只把其他架构模型重命名。现有本机 OCR 文件的精确发布地址/转换链尚未逐个追溯。 |
| 主体分割 | [U-2-Net](https://github.com/xuebinqin/U-2-Net)、[rembg](https://github.com/danielgatis/rembg) | 当前输入为 320×320 的 U2Net ONNX。现有本机转换件的精确版本来源待确认。 |
| 背景修复 | [OpenCV Zoo 的 LaMa](https://github.com/opencv/opencv_zoo/tree/main/models/inpainting_lama) | 对应 `inpainting_lama_2025jan.onnx`；可由 OpenCV 官方仓库的 Git LFS 文件或其官方模型镜像获取。输入 image/mask，输出 RGB 0–255。 |
| 超分辨率 | [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)、[ncnn Vulkan 版本](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan) | 需要官方 Windows 运行文件和 x4plus 模型；老项目更新较少。源码包不提供二进制，运行前应验证官方来源和驱动兼容性。 |

`model-manifest.json` 记录整理时本机既有资源的文件大小与 SHA-256，仅作为这套配置的比对记录。它不是官方发布签名，不证明文件安全或获得再分发授权。文件不匹配时应检查来源、版本和输入输出，不要绕过错误强行加载。

已有本机软件的使用者可以在本机把兼容模型复制到上述位置进行个人验证；公开重新分发模型或软件包前，仍需逐项确认来源和许可。新使用者目前需要自行按上游说明准备兼容资源；由于部分旧资源的精确获取链尚未补齐，本候选版暂不承诺全新电脑一键部署。

模型、上游可执行文件与字体不因应用采用 GPLv3 就自动变成 GPL。模型权重不进入本次源码包。
