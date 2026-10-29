# 第三方组件与资源

应用自行编写的处理流程和工作台采用 GPL-3.0-only。以下依赖、模型和字体保留自身版权及许可。源码包不打包 Python/Node 运行时、模型权重和 Real-ESRGAN 二进制；运行时另行安装的依赖也有各自的传递依赖及声明。

| 项目 | 许可/核对依据 | 用途 |
|---|---|---|
| RapidOCR / PaddleOCR | Apache-2.0；代码许可不代替逐个模型文件核实 | 文字识别 |
| ONNX Runtime | MIT | 本地推理 |
| OpenCV | Apache-2.0（发行包含第三方声明） | 图像处理 |
| U-2-Net / rembg 转换链 | 上游分别为 Apache-2.0 / MIT；当前转换件来源未完全追溯 | 主体分割 |
| LaMa / OpenCV Zoo | Apache-2.0；附本机留存的模型卡与许可 | 补全遮挡背景 |
| Real-ESRGAN | BSD-3-Clause | 超分辨率模型/算法 |
| Real-ESRGAN ncnn Vulkan | MIT，包含 realsr 上游声明 | Windows 增强运行程序，源码包不包含 |
| ag-psd | MIT；注意上游明确排除部分示例图片/笔刷 | PSD 写入 |
| @napi-rs/canvas | MIT 及其第三方声明 | 文字渲染与预览 |
| NumPy / FastAPI / Uvicorn / python-multipart | 以各安装包所附 BSD/MIT/Apache 声明为准 | 计算和本机服务 |
| Pillow | MIT-CMU 及随包声明 | 图像编解码 |
| resvg-py / vtracer | 以各自发行包所附许可为准；留存声明在 third_party | SVG 渲染与轮廓重建 |

附带的五款 `LayerStudioSansSC-*` / `LayerStudioSerifSC-*` 字体由 Noto Sans SC / Noto Serif SC 可变字体生成静态字重并改名，用于避免修改字体沿用保留名称。按 SIL OFL 1.1 提供，许可在 `app/fonts/*-OFL.txt`。字体不是 GPL。原可变字体项目为 [Google Fonts](https://github.com/google/fonts/tree/main/ofl)，本次没有发布字体转换脚本或微软字体。

Windows 自带字体只在本机读取匹配，不随项目再分发。PSD 交付包只复制本项目附带的 OFL 字体。

截至本次核对，部分旧模型精确来源、二进制官方校验链及全部传递依赖审计仍未完成，因此本次只交付源码候选版。`third_party` 中的留存文本用于说明上游许可，不表示把未包含的组件一并授予再分发权限。
