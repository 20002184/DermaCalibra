# DermaCalibra: 基于视觉和元数据的皮肤病变分类系统

## 项目概述

DermaCalibra 是一个基于深度学习的皮肤病变分类系统，结合了图像数据和患者元数据来提高分类准确性。该系统使用改进的 ResNet 模型和注意力机制来处理皮肤病变图像，同时整合患者的临床和人口统计学特征。

## 数据集

### 数据集获取

1. 下载 PAD-UFES-20 数据集：
   - 访问 [PAD-UFES-20 数据集页面](https://data.mendeley.com/datasets/zr7vgbcyr2/1)
   - 点击 "Download" 按钮下载数据集压缩包
   - Patients metadata is preprocessed same as MetaBlock (https://github.com/paaatcha/MetaBlock)
   - 或直接通过我的百度网盘链接下载Metablock处理后的数据链接: https://pan.baidu.com/s/18Fpd9kjF6Bj6J8WfgUIdDA?pwd=wb99 提取码: wb99
2. 数据集存放：
   - 创建项目根目录下的 `PAD-UFES-20` 文件夹
   - 将下载的数据集解压到该文件夹中
   - 确保以下文件结构：
     ```
     PAD-UFES-20/
     ├── imgs/              # 包含所有皮肤病变图像
     ├── pad-ufes-20_parsed_folders.csv   # 训练和验证数据
     └── pad-ufes-20_parsed_test.csv      # 测试数据
     ```

### 数据集结构

项目使用 PAD-UFES-20 数据集，包含以下文件：
- `imgs/`: 包含所有皮肤病变图像
- `pad-ufes-20_parsed_folders.csv`: 训练和验证数据
- `pad-ufes-20_parsed_test.csv`: 测试数据

## 数据划分流程

### 1. 测试集划分（huafen.py）

使用 `huafen.py` 脚本对测试集进行分层划分：

```python
# 安装依赖
pip install pandas numpy scikit-learn

# 运行划分脚本
python huafen.py
```

划分过程：
- 使用 StratifiedShuffleSplit 进行分层抽样
- 保持各子集中诊断类别分布平衡
- 验证划分结果，包括：
  - 诊断类别分布
  - 病灶部位分布
  - 年龄分布
  - 病灶直径分布

### 2. 训练和验证集划分

训练过程中使用 K-fold 交叉验证：
- 将数据集分为 K 个子集
- 每次使用 K-1 个子集作为训练集，剩余一个子集作为验证集
- 通过轮换验证集来评估模型性能

## 模型训练

### 1. 环境配置

```bash
# 安装依赖
pip install torch torchvision tensorboardX pandas numpy scikit-learn
```

### 2. 训练模型

```bash
# 使用默认参数训练
python main.py

# 自定义参数训练
python main.py -cls 6 -batch_size 8 -lr 0.001 -epoch 150 -optimizer sgd -gpu True
```

参数说明：
- `-cls`: 分类类别数（默认：6）
- `-batch_size`: 批量大小（默认：8）
- `-lr`: 学习率（默认：0.001）
- `-epoch`: 训练轮次（默认：150）
- `-optimizer`: 优化器选择（默认：sgd）
- `-gpu`: 是否使用 GPU 加速（默认：True）

### 3. 训练过程监控

- 使用 TensorBoard 记录训练指标
- 保存训练日志到 `logs/` 目录
- 记录以下指标：
  - 训练损失和准确率
  - 验证损失和准确率
  - 测试损失和准确率
  - 平衡准确率（BACC）
  - Top-5 准确率

## 模型测试

```bash
# 测试模型性能
python test_model.py --model_path [模型路径] --batch_size 8 --output_dir [输出目录]
```

测试指标：
- 准确率（Accuracy）
- 平衡准确率（BACC）
- 宏平均 AUC（Macro AUC）

## 模型解释性分析（gfa_analysis.py）

使用梯度特征归因（Gradient Feature Attribution）方法分析模型的决策过程：

```bash
# 运行 GFA 分析
python gfa_analysis.py
```

### 分析内容

1. 图像特征归因：
   - 生成热力图显示模型关注的图像区域
   - 特别关注基底细胞癌（BCC）和鳞状细胞癌（SCC）的诊断特征
   - 可视化结果保存为热力图叠加在原始图像上

2. 元数据特征重要性：
   - 分析各元数据特征对模型决策的影响程度
   - 生成特征重要性柱状图
   - 帮助理解模型如何利用患者临床和人口统计学特征

### 输出结果

分析结果保存在 `pad_model/gfa_results/` 目录下：
- `gfa_heatmap_sample_X_class_Y.png`: 图像特征归因热力图
- `gfa_metadata_sample_X_class_Y.png`: 元数据特征重要性图
- `gfa_analysis.log`: 分析过程日志

## 模型保存规则

模型在满足以下条件时会被保存：
1. 验证集性能：
   - 准确率 > 0.83
   - BACC > 0.805
2. 测试集性能：
   - 准确率 > 0.848
   - BACC > 0.811
   - 宏平均 AUC > 0.96

保存的模型文件命名格式：
```
[model_name]_[accuracy]_[bacc]_[macro_auc]_epoch_[epoch_number].pkl
```
## 测试合成 HAM10000 和 ISIC 2019 数据集
# 测试 HAM10000 数据集
```bash
 python test_ham.py --model_path [模型路径] --batch_size 8 --output_dir [输出目录]
```
 ```bash
# 测试 ISIC 2019 数据集
 python test_isic.py --model_path [模型路径] --batch_size 8 --output_dir [输出目录]
```
## 输出文件

- 训练日志：`logs/[timestamp]_training.log`
- 预测结果：`pad_model/prediction_[model_name]_[timestamp].csv`
- 训练指标：`pad_model/epoch_[model_name].csv`
- TensorBoard 日志：`log/[timestamp]/`
- GFA 分析结果：`pad_model/gfa_results/`

## 注意事项

1. 确保数据集路径正确配置
2. 建议使用 GPU 进行训练
3. 定期检查训练日志，监控模型性能
4. 保持足够的磁盘空间用于保存模型和日志
5. GFA 分析需要较大的计算资源，建议使用 GPU 运行








