# PDM-Net：可运行的研究实现

此目录按用户指定命名为 `PDE-Net`，实现的模型是 Huang 等（IJCAI 2024）提出的 **PDM-Net**，论文为 *Long Short-Term Dynamic Prototype Alignment Learning for Video Anomaly Detection*。

已实现模型、正常视频数据采样、两步交替训练、损失、检查点和离线异常评分。本文档记录可运行实现的具体选择；**不是作者官方代码，也尚未复现论文的基准 AUC**。

## 安装和快速验证

Python >= 3.9。依赖为 PyTorch、NumPy、Pillow。以下命令在本目录执行：

```bash
cd /Users/lyw/Desktop/code/PDE-Net
python3 -m pip install -r requirements.txt
python3 -m pdm_net smoke
python3 -m unittest discover -s tests -v
```

`smoke` 使用随机数据和小模型执行一次完整交替训练及推理，只验证程序能运行。默认完整模型配置在 `configs/default.json`；`configs/smoke.json` 只用于调试。

`--device auto` 自动选择 CUDA，否则使用 CPU。macOS 的 MPS 可用 `--device mps` 显式指定，但当前实现仅验证了 CPU；不同 PyTorch/macOS 版本对 Conv3d 等算子的支持不同。

## 目录与已实现模块

```text
pdm_net/
  config.py                  配置及参数校验
  data/dataset.py             视频内采样、自然排序、目标帧标签对齐
  models/
    residual.py              有符号/绝对帧差
    encoders.py              空间编码器、独立长/短期3D动态编码器
    convlstm.py              ConvLSTM单元及多层时序编码
    prototype.py             原型读、最相关原型、显式写入
    aggregation.py           原型与记忆状态的通道注意力融合
    decoder.py               RGB目标帧解码
    pdm_net.py               完整模型及长/短分支接口
  losses.py                  预测、紧凑、分散损失
  engine/trainer.py          每批两步训练、优化器、检查点恢复
  engine/evaluator.py        PSNR、匹配误差、逐视频归一化、AUC
  cli.py                     smoke / train / evaluate 命令
configs/                     完整及小尺寸配置
tests/test_pipeline.py       数值、梯度、采样、状态恢复等测试
ARCHITECTURE.md              本实现的模型图、张量接口与梯度归属
```

## 数据准备

输入为已拆帧的视频，每个视频独立一个文件夹，帧文件按自然数字排序。支持 PNG、JPG、JPEG、BMP、TIF、TIFF；图片转换为 RGB，缩放到配置尺寸，数值归一化到 `[0,1]`。当前不直接解码 MP4/AVI，也不解析特定基准数据集的 `.mat` 标注。

```text
data/
  train/
    video001/  0.png  1.png  ...
    video002/  0.png  1.png  ...
  test/
    video003/  0.png  1.png  ...
  labels/
    video003.npy
```

训练目录必须只包含正常视频。若原视频含异常，应先按连续正常片段拆分，不能删除异常帧后把前后帧直接拼接。本实现不自动识别训练视频中的异常。

每个标签文件是长度等于该视频帧数的一维 NumPy 数组：`0=正常，1=异常`。标签下标对应自然排序后的帧位置，不是文件名中的数字。评估可不提供标签，此时只导出分数，AUC 为 `null`；只有一种类别的 AUC 也记为 `null`。

默认采样窗口为14帧：长输入 `0..12`、短输入 `0..8`、目标 `13`。窗口每次平移一帧，始终处于同一个视频内。评估仅读取短输入和目标，不读取隐藏的中间帧；前13帧无足够上下文，**不生成分数，也不补零参与 AUC**。不足14帧的视频会警告并跳过。

## 训练、恢复和评估

```bash
python3 -m pdm_net train \
  --data /absolute/path/data/train \
  --config configs/default.json \
  --output runs/experiment1 \
  --epochs 50 --batch-size 1 --device auto

python3 -m pdm_net train \
  --data /absolute/path/data/train \
  --resume runs/experiment1/latest.pt \
  --output runs/experiment1 \
  --epochs 100 --batch-size 1 --device auto

python3 -m pdm_net evaluate \
  --data /absolute/path/data/test \
  --labels /absolute/path/data/labels \
  --checkpoint runs/experiment1/latest.pt \
  --output runs/experiment1/evaluation --device auto
```

`--epochs` 指总轮数；从50轮恢复并指定100时再训练50轮。每轮保存 `latest.pt`、追加 `metrics.jsonl`，模型配置另存为 `config.json`。检查点包含网络、原型、两个 Adam 优化器、完成轮数、迭代数及 PyTorch 随机状态。恢复以检查点配置为准，额外传入不一致的 `--config` 会报错。支持轮末恢复，不支持轮内断点恢复。若未指定恢复且输出位置已有 `latest.pt`，会拒绝覆盖。

评估输出 `scores.csv`（视频、目标索引、文件名、PSNR、匹配误差、可选标签、异常分数）和 `metrics.json`（整体帧级 AUC、逐视频 AUC、有效帧数）。AUC 范围为 `[0,1]`。两种原始分数均按**同一视频的全部有效目标帧**归一化后再融合，必须等该视频评分完成；这不是在线实时分数。

## Python 接口

```python
import torch
from pdm_net import PDMConfig, PDMNet
from pdm_net.engine import AlternatingTrainer

model = PDMNet(PDMConfig())
trainer = AlternatingTrainer(model)
long_clip = torch.rand(1, 13, 3, 256, 256)
batch = {
    "long_clip": long_clip,
    "short_clip": long_clip[:, :9],
    "target": torch.rand(1, 3, 256, 256),
}
metrics = trainer.train_batch(batch)

model.eval()
with torch.no_grad():
    output = model(batch["short_clip"])
    predicted_frame = output.prediction  # [1,3,256,256]
```

`model(short_clip)` 是推理/第二步接口；`model.forward_long(short_clip,long_clip)` 是第一步接口。两者都不接受目标帧作为预测输入，且前向传播不会隐式写入原型。直接调用训练接口时，调用方负责保证长短片段属于同一窗口、目标在正确位置、输入为 `[0,1]`。

## 论文公式与实现约定

已实现的核心关系：

- 读权重：局部查询和原型的余弦相似度，沿原型维 softmax，再加权召回全部原型。
- 写权重：沿查询维 softmax，在各原型分配集合内除以最大权重，再按公式5加权写入并 L2 归一化。没有分配查询的原型保持不变。
- 预测损失：平方误差与绝对误差之和。
- 紧凑损失：查询到余弦最相关单个原型的平均 **非平方 L2 距离**。
- 分散损失：按查询分配频率加权的原型类间散度矩阵的负迹。
- 第一步：`L_pre + 0.1 * (L_com + L_dis)`；第二步：`L_pre + 0.1 * L_com`。
- 评分：`0.6 * (1 - normalize(PSNR)) + 0.4 * normalize(matching_distance)`。

需要与作者实现进一步核对的选择：

| 项目 | 本实现的选择 | 影响 |
|---|---|---|
| 卷积结构 | 空间编码器4层2D卷积；动态编码器4层3D卷积，仅空间下采样，最后时间平均 | 论文正文没有给出完整层配置，不能视为官方网络 |
| 隐状态与融合 | 2层ConvLSTM，取顶层最后时刻 H/C；拼接原型/C后全局池化+通道MLP | 层数和具体MLP为工程配置 |
| 通道与解码 | 默认32基础通道、256特征通道；4次双线性上采样+卷积，Sigmoid输出 | 可通过配置调整宽度 |
| 残差 | 默认有符号 `x[t+1]-x[t]`，可选 absolute | 论文未充分说明符号处理 |
| 查询归一化 | 余弦相似度内归一化；损失和显式写入使用原始查询 | 特征幅值会影响匹配误差和写入 |
| 预测损失缩放 | 默认逐元素平均；可选 sum_per_sample（每样本求和后对批平均） | 与紧凑/分散项的相对尺度不同，需消融 |
| 采样 | 13帧长输入、9帧短输入、索引13为目标 | 论文短序列端点有计数歧义，暂按声明长度实现 |
| 批内写入 | 将整个批次 B*h*w 个查询视为一个集合 | 论文以单样本 K=h*w 表述；批聚合为工程约定 |
| 原型更新 | 分散梯度更新并归一化，然后按公式5显式写入 | 正文与算法的梯度归属不完全一致，采用下述明确策略 |

**原型梯度策略：** 原型是可训练 Parameter，但读操作对原型执行 `detach()`，不切断查询梯度。因此预测/紧凑损失只更新网络；分散损失单独向原型传播。第一步先更新网络，再用独立 Adam 更新原型分散项并归一化，最后使用本次长期前向得到的查询，在更新后的原型上重新计算分配并显式写入。第二步只更新短期编码器和共享预测器，原型完全固定。这保证分散项不是没有梯度的装饰项，但**该更新顺序仍是实现约定，不能宣称为严格复现**。

数值边界：零 MSE 用 `1e-12` 截断，最大 PSNR 为120；常数序列的 min-max 输出零。因此常数预测项贡献固定的0.6，常数匹配项贡献零，常数视频内没有排序能力。异常分数不是校准概率，不建议直接用固定0.5阈值解释。

## 当前验证范围

- 12项 CPU 单元测试覆盖原型读写公式、非平方紧凑损失、分散梯度、两步参数更新边界、推理只读、采样与标签对齐、评分方向/常数、检查点下一步精确恢复等。
- 小模型随机数据完成训练一轮、加载检查点继续一轮、评估导出 CSV/JSON 的命令行流程。
- 默认完整配置验证9帧256×256输入，输出 `[1,3,256,256]`，原型权重 `[1,256,100]`。
- 未进行真实数据集训练；未验证 CUDA/MPS、论文AUC、实时速度或跨数据集泛化。
