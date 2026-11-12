# 来源与改动说明

## 检测模型

- BlurBall：<https://github.com/cogsys-tuebingen/blurball>
- 固定源码提交：`2f0f5496f7ba4b5b1a36790749935121b2ce972d`
- 原作者：Thomas Gossard、Filip Radovic、Andreas Ziegler、Andreas Zell。
- 原始模型实现位于 `pingpong/vendor/blurball.py`，原样保留；MIT 许可证位于同目录的 `BLURBALL_LICENSE.md`。文件中上游 HRNet 的版权说明保留。
- 权重使用作者 README 指向的公开 Nextcloud：<https://cloud.cs.uni-tuebingen.de/index.php/s/6Z8TpM3sXRKHzGC>，文件 `blurball_best`。
- 参考训练素材来自作者 README 指向的公开数据：<https://cloud.cs.uni-tuebingen.de/index.php/s/C3pJEPKWQAkono7>，00 和 01 目录各 12 段及配套 midpoint 标注。数据授权须结合原作者说明核对；代码 MIT 许可不能自动替代数据使用条件。
- 本项目新增独立模型加载、RGB 归一化、ONNX 导出数值核验、流式逐帧结果对齐、连通区域球心提取、显卡/CPU 推理选择、训练和评测记录。

## 自主实现部分

中文桌面交互、非阻塞分析与取消、项目保存和恢复、逐帧与慢速回放、手动球台标定、触台候选规则、候选复核与补充、球心修正、桌面落点与热力图、CSV/JSON/MP4 导出、数据审计、回归检查及 Windows 打包流程。

## 技术参考

- TTNet 非官方实现：<https://github.com/maudzung/TTNet-Real-time-Analysis-System-for-Table-Tennis-Pytorch>。用户下载的源码完整保留，仅作技术参考，发布软件未复制其中源码，也没有使用未训练的 TTNet 权重。
- TT3D：<https://github.com/cogsys-tuebingen/tt3d>。v2 使用其 `weights/table_segmentation.ckpt` 预训练台面分割权重，结构为 UnetPlusPlus / timm-regnety_002，导出为 `models/table.onnx`，不依赖 Lightning 在桌面端运行。原许可证保存在 `pingpong/vendor/TT3D_LICENSE.txt`，并随软件分发。当前没有集成其三维重建算法。
- Segmentation Models PyTorch：<https://github.com/qubvel-org/segmentation_models.pytorch>。在模型转换环境中重建与权重严格匹配的网络。运行软件仅使用转换后的 ONNX，不安装该训练库。
- PySide6 / Qt for Python：桌面界面；ONNX Runtime DirectML：部署推理；PyAV / FFmpeg：主要视频解码、实际时间戳与 H.264 编码；OpenCV：备用解码、绘制及平面几何；NumPy / SciPy：数值处理。PyAV 及其二进制依赖许可证随独立版分发。

模型来源、原有功能和新增功能应在参赛报告中分别说明，不能将整个检测主干声明为原创模型。

## 本轮新增训练数据

- ExtendedOpenTTGames：<https://github.com/moamal01/table_tennis_data>，由用户提供本地标注文件。保留其原始 JSON，不将缺少标签的帧当作无球负样本。
- 原始视频和球坐标来自 OpenTTGames：<https://lab.osai.ai/>。训练准备脚本通过官方视频时间戳按需解码对应帧，训练、验证和测试按完整视频来源分组。
- 数据集许可为 CC BY-NC-SA 4.0，说明保存在 assets/ExtendedOpenTTGames_LICENSE.txt 并随软件分发。微调产物应保留数据来源及相应授权条件，不能用检测主干的 MIT 代码许可证代替数据许可。
- 普通碰网事件与回合结束事件采用不同标签，不能把所有 net 都当成回合终止。本轮训练和指标的具体范围见 reports/extended 及 runs/extended-transfer-*，小球定位分数不代表触台、出界或拦网分数。
