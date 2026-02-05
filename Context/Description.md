# 模块说明（按文件）

## 依赖库（Pipeline 必需/可选）
- 必需：`numpy`, `torch`, `h5py`
- 可选（可视化）：`matplotlib`
- 可选（滤波）：`scipy`
- 可选（Excel 读取/导出）：`pandas`, `openpyxl`

## Main/S1/ingest.py
### 自定义类
- `ChannelSeries`：单个通道的数据载体，包含通道名、数值序列和来源文件路径。
- `RawSignal`：原始信号对象，保存 `C×1×T` 的数据矩阵、通道名和来源文件列表，并提供 `shape` 便捷属性。
- `WaveformEvents`：候选事件波形对象，保存 `X(C×N×t)` 与 `T0(C×N)`，并可记录通道名与来源。
- `SpikeInput`：输入封装类型，标记输入是 `raw` 还是 `wave`，并对应保存解析结果。

### 主要函数
- `load_input`：根据 `data_type` 调用 `load_raw` 或 `load_wave`，统一返回 `SpikeInput`。
- `load_raw`：读取 Excel 文件（sheet=channel）并组装为 `RawSignal`。
- `load_wave`：读取 X/T0 两个 Excel 文件（sheet=channel），组装为 `WaveformEvents`。

### 关键辅助函数
- `_collect_input_files`：收集指定目录下的 Excel 文件。
- `_find_wave_pair`：在目录中自动匹配 X 与 T0 文件。
- `_read_wave_X` / `_read_wave_T0`：读取波形与时间点，检查尺寸。
- `_ensure_same_shape` / `_ensure_same_length`：保证多通道数据一致。
- `_validate_wave_shapes`：校验 X 与 T0 的形状对齐。
- `_merge_channel_names`：合并/校验通道名一致性。
- `_read_excel_channels`：读取 Excel 所有 sheet 并转换为通道序列。

## Main/S1/__init__.py
- 作为模块导出文件，集中暴露 S1 的主要类与函数。

## Main/S2/preprocess.py
### 自定义类
- `FilterResult`：滤波结果对象，保存滤波后的数据与参数（截止频率、采样率、滤波器类型、阶数）。

### 主要函数
- `select_channels`：按索引或名称筛选通道。
- `plot_raw`：可视化指定通道区间的原始波形（需要 matplotlib）。
- `highpass_filter`：高通或带通滤波（基于 Butterworth + filtfilt）。

### 辅助函数
- `_normalize_index`：处理负索引并做范围校验。
- `_unique_ordered`：去重并保持原有顺序。

## Main/S2/__init__.py
- 作为模块导出文件，集中暴露 S2 的主要类与函数。

## Main/S3/candidate.py
### 自定义类
- `CandidateResult`：候选事件检测结果，包含 `X/T0`、每通道数量、阈值以及窗口参数。

### 主要函数
- `pick_threshold`：多通道可视化阈值选择器（每通道独立滑块），返回可用于 `detect_candidates` 的阈值列表。
- `detect_candidates`：负峰阈值穿越检测，按 `pre/post` 窗口切片生成 `X/T0`。

### 辅助函数
- `_expand_thresholds`：将单值阈值扩展为逐通道阈值列表。
- `_find_negative_crossings`：负阈值穿越检测。
- `_default_threshold`：默认阈值=全局最小值的一半。
- `_normalize_channel_index`：通道索引归一化（支持负索引）。

## Main/S3/__init__.py
- 作为模块导出文件，集中暴露 S3 的主要类与函数。

## Main/S4/train.py
### 自定义类
- `TrainData`：训练数据载体，包含 `X(N×T)` 与 `y(N)`。
- `WaveformDataset`：PyTorch Dataset 封装，提供 `__len__` / `__getitem__`。

### 主要函数
- `load_training_data`：读取 Excel 格式训练数据（X 与 label 两个文件）。
- `build_simple_mlp`：构建一个简单的 MLP 分类模型。
- `train_model`：训练循环（交叉熵 + Adam）。
- `save_model`：保存模型参数到文件。

### 辅助函数
- `_find_train_pair`：在目录中自动匹配 X 与 label 文件。

## Main/S4/__init__.py
- 作为模块导出文件，集中暴露 S4 的主要类与函数。

## Main/S5/predict.py
### 主要函数
- `predict`：批量推理，输出每个样本的类别索引（argmax）。

## Main/S5/__init__.py
- 作为模块导出文件，集中暴露 S5 的主要函数。

## Main/S6/exporter.py
### 自定义类
- `ExportBundle`：导出数据集合，按类别保存 `WaveformEvents`。

### 主要函数
- `split_by_class`：根据 `pred` 划分波形到不同类别。
- `export_to_excel`：将分好类的波形保存为 Excel（每类 X/T0 各一个文件）。

### 辅助函数
- `_pad_channel_events`：对不同通道的事件数进行 padding 对齐。
- `_write_wave_excel`：写出 Excel 文件（sheet=channel）。

## Main/S6/__init__.py
- 作为模块导出文件，集中暴露 S6 的主要类与函数。

## Main/S7/qc.py
### 自定义类
- `QCReport`：QC 报告对象，包含候选数、类别统计、p2p 与 SNR 统计信息。

### 主要函数
- `generate_qc_report`：生成 QC 统计结果。

### 辅助函数
- `_peak_to_peak`：计算峰-峰值。
- `_simple_snr`：计算简化 SNR。
- `_summary_stats`：统计均值、中位数、90 分位数。

## Main/S7/__init__.py
- 作为模块导出文件，集中暴露 S7 的主要类与函数。
