# 一、设计目标

在强噪声、强 伪迹的在体电生理数据中，通过深度学习对候选Spike进行提取，并允许用户用自己的标注数据训练专属模型。

# 二、流程


1) 输入层 Ingest
输入文件夹：./Data/Spike data
输入类型：.mat

支持两类输入：
Type-1: raw data：宽带连续信号 raw，格式为C * 1 * T ，C为通道，T为时间

Type-2: waveform：候选事件波形 wave，格式为{X[C,N,t], T0[C,N]}。C为通道，N为Spike的数目，t为时间窗口长度，T0为spike初始时间点


2) 预处理 Preprocess（仅对 Type-1）

输入：raw

滤波：高通滤波200Hz
参考：CAR / CMR（可选，通道数<2时无法使用）
坏道处理：可手动标注坏的通道


I/O
Input: raw，格式为C * 1 * T ，C为通道，T为时间
Output: hp_signal，格式为格式为C * 1 * T ，C为通道，T为时间


3) 候选检测 Candidate Detection（仅对 Type-1）

目标：高召回，不漏；宁可多候选。

阈值：thr_ch (可视化hp_signal后通过一个可移动的阈值线设置)
峰值规则：负峰（固定）
窗口长度：t1为与阈值线相交的时间点，窗口长度为[t1 - pre, t1 + post]

I/O
Input: hp_signal
Output: wave，格式为{X[C,N,t], T0[C,N]}。C为通道，N为Spike的数目，t为时间窗口长度，T0为spike初始时间点


4) 模型训练

数据集:waveform dataset. X_train[N,t] label[N] ，N为Spike的数目
数据集文件夹：./Data/Train data
模型选择：mlp, resnet
Loss：
优化器：


模型储存
储存位置：./Model/



5) 模型输出 Spike(Multi-class Classifier)

输入：X[C,N,t]
输出：pred[C,N]


7) 导出 Export

每个类别的{X.class[C,n,t], T0.class[C,n]}


8) QC 报告 QC Report

净化前 vs 净化后对比（最能证明价值）：
候选数量下降比例
spike 类占比随时间稳定性
波形 SNR / p2p 分布变化
ISI violation、unit stability 改善




















