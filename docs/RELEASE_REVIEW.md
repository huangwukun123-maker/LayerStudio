# 开源候选版检查记录

检查日期：2026-10-21。应用版本：0.3.5-oss.1。应用原创源码许可：GPL-3.0-only。

## 已完成的整理

- 从现有 0.3.5 源码白名单复制核心，在独立目录移除激活页面、授权接口、联网核验和管理配置。
- 保留本机监听、Host/Origin 限制、上传大小/数量/像素限制、任务队列及下载白名单。
- 修复并发上传可突破 30 项待处理上限的问题，新增回归测试；上传失败也会释放预留额度。
- 字体仅随交付包附带，不自动安装，不写字体注册表。修正启动器使用任务目录环境变量的行为。
- 补齐 resvg-py、vtracer 运行依赖；Node 依赖固定版本。15 个 Node 锁文件条目的 integrity 与官方 npm 发行元数据一致，下载地址改为 registry.npmjs.org。
- 旧 Pillow 12.2.0 的 PyPI 数据返回已知安全公告；改为 Pillow 13.0.0。验证使用官方 PyPI 元数据所列 Windows CPython 3.12 wheel，SHA-256 一致，在独立工作目录加载，未替换原软件环境。13.0.0 本次查询未返回已知公告，不等于无漏洞。依据：[官方发布说明](https://pillow.readthedocs.io/en/stable/releasenotes/13.0.0.html)、[PyPI 版本元数据](https://pypi.org/pypi/Pillow/13.0.0/json)。
- 源码包排除原有任务、客户素材、授权记录、后台配置、运行缓存、模型、运行环境与第三方可执行文件。依赖私有图片或历史字标样本的测试没有纳入；通用渐变背景回归改用合成图片。

## 验证结果及实际边界

| 验证 | 结果 |
|---|---|
| Python 公开测试（含无激活、并发上限、字体导出、文字/背景/透明边缘） | 95 通过 |
| Node PSD 语义与前端展示测试 | 8 通过 |
| 实际启动器 | 无激活进入工作台、报告缺失模型、使用指定任务目录、正常退出 |
| 本机既有 ONNX 模型 | 合成图识别到 SALE 2026；主体分割及 LaMa 背景修复成功；蒙版外像素保持不变 |
| 旧 Vulkan 增强 EXE | 未执行，官方二进制来源校验链未补齐 |
| 从图片到 PSD 的完整真实模型流水线 | 本次未完整重跑；PSD 写入由独立测试验证 |
| 全新电脑安装、不同显卡和其他系统 | 未验证 |

Python 测试使用已有 Python 3.12 环境及隔离加载的 Pillow 13.0.0；Node 测试复用本机已有依赖。未声称完成全新环境安装。测试有一项来自 Starlette/httpx 的弃用提示，测试全部通过。本次没有执行未知安装器、改变系统安全设置或启动原授权后台。

## 上游维护与许可核对

下表来自 GitHub 公开仓库 API 的本次快照；贡献者是前 30 名样本中区分 User/Bot 后的人数，**不是总贡献者数**。最后提交使用默认分支提交时间；更新时间与代码提交时间分别查看。Issue/PR 各抽样最近 3 项及状态，未逐项审计全部历史。Star/Fork 不能证明安全。

| 上游 | Star | Fork | 样本人类/机器人 | 默认分支最近提交 | 许可 |
|---|---:|---:|---:|---|---|
| [RapidAI/RapidOCR](https://github.com/RapidAI/RapidOCR) | 8247 | 754 | 29/1 | 2026-10-20T14:06:30Z | Apache-2.0 |
| [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | 91469 | 11536 | 30/0 | 2026-09-16T03:30:53Z | Apache-2.0 |
| [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) | 22129 | 4346 | 28/2 | 2026-10-21T08:45:56Z | MIT |
| [xuebinqin/U-2-Net](https://github.com/xuebinqin/U-2-Net) | 9883 | 1615 | 10/1 | 2024-01-13T04:57:35Z | Apache-2.0 |
| [danielgatis/rembg](https://github.com/danielgatis/rembg) | 25145 | 2444 | 30/0 | 2026-10-10T17:17:20Z | MIT |
| [advimman/lama](https://github.com/advimman/lama) | 10331 | 1085 | 13/0 | 2025-02-05T08:41:50Z | Apache-2.0 |
| [opencv/opencv_zoo](https://github.com/opencv/opencv_zoo) | 1097 | 313 | 30/0 | 2026-10-20T15:55:38Z | Apache-2.0 |
| [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | 37166 | 4492 | 11/0 | 2024-04-02T16:39:11Z | BSD-3-Clause |
| [xinntao/Real-ESRGAN-ncnn-vulkan](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan) | 2269 | 303 | 2/0 | 2022-04-24T08:46:32Z | MIT（已读 LICENSE） |
| [Agamnentzar/ag-psd](https://github.com/Agamnentzar/ag-psd) | 731 | 93 | 10/0 | 2026-10-07T19:07:13Z | MIT（已读 LICENSE） |
| [Brooooooklyn/canvas](https://github.com/Brooooooklyn/canvas) | 2342 | 100 | 27/3 | API 限流，未确认 | MIT + 第三方声明（已读 LICENSE） |
| [google/fonts](https://github.com/google/fonts) | 20659 | 2934 | 29/1 | API 限流，未确认 | 按字体分别授权；本项目附带字体 OFL 1.1 |

Pillow 另外核对官方仓库页及 13.0.0 官方发布说明：页面显示约 13.9k Star、2.5k Fork、64 个 Issue、71 个 PR；贡献者精确人数与最新提交时间本次未确认。它是长期维护的多人项目，但仍有持续公布的解码漏洞，因此本候选版使用已验证的新版本。参考：[Pillow 仓库](https://github.com/python-pillow/Pillow)。

观察：U-2-Net、原始 LaMa、Real-ESRGAN 和 Vulkan 包的代码提交较旧，Vulkan 包人类贡献者样本仅 2 人。这些既有依赖没有因 Star 较高就被判定为安全；候选版不重新分发或运行来源未核实的旧增强二进制。canvas/google/fonts 部分提交信息遇公开 API 限流，已标记未确认。

完整上游元数据及 Issue/PR 抽样见 [upstream-audit.json](upstream-audit.json)。README 描述与本项目实际调用路径、输入输出进行了对应核对；未逐行审计全部上游源码。上游 MIT 原文和字体 OFL 声明已留存。

## 未完成、不能据此宣称的事项

- 现有 OCR/U2Net 转换件的精确下载地址和转换版本尚未逐个追溯，旧 Vulkan 二进制的官方哈希链也未完成；模型和二进制不纳入本次公开源码包。
- Python 仅固定直接依赖；全部传递依赖的锁定、漏洞审计、二进制构建链和签名核实未完成。npm integrity 比对只证明对应发行文件的一致性，不是源代码审计或安全证明。
- 全新电脑的完整模型准备链尚未验证，因此目前为源码候选版，不能宣传一键安装或全流程实测通过。
- 根目录 GPLv3 不替代第三方模型、库、字体和商标的许可；未来再分发完整程序或权重需进一步核实。

## GitHub 发布状态

2026-10-29 已从用户指定的夸克浏览器可见页面确认发布账号为 `huangwukun123-maker`，拟使用仓库名 `LayerStudio`。发布前通过 GitHub 官方设备授权连接正确账号，不提取浏览器 Cookie 或使用原有其他账号。本文是发布前检查记录，仓库实际状态以 GitHub 页面为准。源码包不包含 Git 凭据。

2026-10-29 追加检查：重新读取当前 10 项直接 Python 依赖对应版本的官方 PyPI 元数据，均未返回已知漏洞公告；15 项 npm 锁文件条目 integrity 再次与官方 npm 元数据核对一致。这不是全部传递依赖审计，也不把 2026-10-21 的模型测试描述成在 10 月 29 日重新完成。
