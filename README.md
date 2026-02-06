# Deep-learning-based-spike-purification-and-classification
# 基于深度学习的在体多通道记录的spike候选提取


本pipeline主要用于解决有信噪比低或存在大量噪音的在体多通道记录的动作电位分选问题


在信噪比低/大量噪音的前提下，Spike的波形往往被掩盖在大量的随机噪声中。由于噪音的随机分布，在特征空间中通常会形成一个大聚团，从而使得基于距离的传统分类方法失效。

![微信截图_20260206102725](https://github.com/user-attachments/assets/d252880e-d98d-4f8f-b666-ce51f40f0c97)

在这种情况下，可以通过自己的之前记录的信噪比高的信号作为训练数据（尤其是同一只动物在不同条件下），通过1d卷积核提取波形特征，来实现对动作电位的直接抓取。

训练完后通过模型对有噪音的记录数据Spike进行提取或sorter

![微信截图_20260206103443](https://github.com/user-attachments/assets/40ab5a3c-0674-475b-922b-425eb13fa218)

具体Pipeline的流程可以参考Notebook文件夹

具体的API接口可以参考Context文件夹

如果自身数据集较小无法训练，可以选择网络上的公开数据集，目前采用的数据集来源为：
https://crcns.org/data-sets/hc/hc-3/about-hc-3
Mizuseki, K., Sirota, A., Pastalkova, E., Diba, K., Buzsáki, G. (2013)
Multiple single unit recordings from different rat hippocampal and entorhinal regions while the animals were performing multiple behavioral tasks. CRCNS.org.
http://dx.doi.org/10.6080/K09G5JRZ











