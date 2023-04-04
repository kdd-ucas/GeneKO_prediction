# GeneKO_prediction

data_process
==
data_merge.py:将原始数据整合成基因表达矩阵
-
data_plot.py:对数据进行画图分析
-
seq_process.py:将dna序列文件处理成onehot向量
-


model
=
data_process.py:用已知的基因调控网络关系建图
-
model_final.py:对基因敲除进行预测
-
get_acc.py:对结果进行评价，画出混淆矩阵
-
net.py:网络结构
-
