import os.path as osp
import pandas as pd
import torch
import torch.utils.data as data
import torch.nn as nn
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from mpl_toolkits.axes_grid1 import host_subplot
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data, download_url, extract_gz
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
from get_acc import tag_test,heat_map,heat_map_acc
from net import *
from sklearn.metrics import confusion_matrix
from data_process import load_data
import time
from plot_matrix import plot_acc_loss
from data_process import get_fre_5
from data_process import get_fre_3

time_start = time.time()
device = torch.device("cuda:3")
# device = torch.device('cpu')
transform = T.Compose([
    T.NormalizeFeatures(),
    T.ToDevice(device),
    T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                      add_negative_train_samples=False),
])


data_path = "/home/cuiwentao/cwt/gene/GAE/tf_gene.csv"
# data_path = "/home/cuiwentao/cwt/gene/model_multi_task/tf_gene_2424.csv"
dataset, gene_mapping = load_data(data_path)
print("Number of genes:", len(gene_mapping))

train_data, val_data, test_data = transform(dataset)
print("Train Data:", train_data)
print("Validation Data:", val_data)
print("Test Data:", test_data)


criterion1 = torch.nn.BCEWithLogitsLoss().to(device)
def criterion2(a,b,fre):
    loss = torch.nn.SmoothL1Loss(reduction='mean').to(device)
    loss =loss(a,b)

    loss = loss * fre
    loss = torch.mean(loss)
    loss = torch.sqrt(loss)
    return loss
criterion3 = torch.nn.MSELoss().to(device)


eval_loss_list = []
eval_acc_list = []
 #8069
# data1 = pd.read_csv("/home/cuiwentao/cwt/mode-feature102.csv")
data1 = pd.read_csv('/home/cuiwentao/cwt/gene/tf_ko108.csv')
feature_ko = pd.read_csv('/home/cuiwentao/cwt/gene/feature_ko_623_108.csv')
feature_ko = feature_ko.values.T
data1 = data1.drop(['index'], axis=1)  # 去除第一列索引)

num_gene = data1.shape[0]
num_gene_double = int((data1.shape[0])*2)
num_sample = data1.shape[1]
num_sample_c = int((data1.shape[1])/2)

data1 = np.log(data1 + 1)

data_c = data1.iloc[:, num_sample_c:].values.T
data_t = data1.iloc[:, :num_sample_c].values.T

target_fre = get_fre_3(data_c,data_t)

data_c=np.concatenate((data_c,feature_ko),axis=1)
data_c=np.concatenate((data_c,target_fre),axis=1)

train_c, test_c, train_t, test_t = train_test_split(data_c, data_t)

data_c_train = torch.tensor(train_c, dtype=torch.float32)
data_t_train = torch.tensor(train_t, dtype=torch.float32)
data_c_test = torch.tensor(test_c, dtype=torch.float32)
data_t_test = torch.tensor(test_t, dtype=torch.float32)

datac1 = data_c_train[:, 0:num_gene].to(device)
datac1 = torch.tensor(datac1, dtype=torch.float32)

torch_dataset = data.TensorDataset(data_c_train, data_t_train)
torch_dataset1 = data.TensorDataset(data_c_test, data_t_test)

loader = data.DataLoader(
    dataset=torch_dataset,
    batch_size=512,  # 每批提取的数量
    shuffle=True,  # 要不要打乱数据（打乱比较好）
    num_workers=2  # 多少线程来读取数据
)
loader1 = data.DataLoader(
    dataset=torch_dataset1,
    batch_size=512,  # 每批提取的数量
    shuffle=True,  # 要不要打乱数据（打乱比较好）
    num_workers=2  # 多少线程来读取数据
)


model = NET(dataset.num_features, 256, 256,n_feature=num_gene_double, out=num_gene).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

def train(batch_x, batch_y):
    model.train()
    # print(batch_x.shape)
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)

    z = model.encode(train_data.x, train_data.edge_index)
    # We perform a new round of negative sampling for every training epoch:

    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

    edge_label_index = torch.cat(
        [train_data.edge_label_index, neg_edge_index],
        dim=-1,
    )
    edge_label = torch.cat([
        train_data.edge_label,
        train_data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)

    # edge_label_index = train_data.edge_label_index
    # edge_label = train_data.edge_label
    # print(edge_label_index)
    out = model.decode(z, edge_label_index).view(-1)
    batchx = batch_x[:, 0:num_gene_double]
    fre = batch_x[:, num_gene_double:]
    # print(fre.shape)
    encode_vector, prediction = model.decode_re(batchx)  # 用网络预测一下
    out2 = model.enco(z, encode_vector)
    loss1 = criterion1(out, edge_label)
    loss2 = criterion2(prediction, batch_y,fre)
    loss3 = criterion2(out2, datac1.t(),fre.t())
    # loss3 = criterion3(out2, datac1.t())
    loss = loss1 + loss2 + loss3
    print(f'loss: {loss:.4f}, Loss1: {loss1:.4f},loss2: {loss2:.4f}, '
          f'loss3: {loss3:.4f}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())

eval_loss_list = []
eval_acc_list = []


for epoch in range(1, 500):
    best_val_acc = final_test_acc = 0
    for batch_idx, (batch_x, batch_y) in enumerate(loader):
        loss = train(batch_x, batch_y)
        val_acc = test(val_data)
        test_acc = test(test_data)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            final_test_acc= test_acc
        eval_loss_list.append(loss)
        eval_acc_list.append(final_test_acc)

    if epoch%10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f},Val: {val_acc:.4f}, '
              f'Test: {test_acc:.4f}')
        # train_auc = test(train_data)



@torch.no_grad()
def test1(loader1):
    best_val_auc = final_test_auc = 0
    model.eval()
    for step, (batchx, batchy) in enumerate(loader1):
        batchx = batchx.to(device)
        batchy = batchy.to(device)

        batchx1 = batchx[:, 0:num_gene_double]
        x1,x2 = model.decode_re(batchx1)
        res = tag_test(batchx1, x2, num_gene)
        res_cont = tag_test(batchx1, batchy, num_gene)
        val_auc = test(val_data)
        test_auc = test(test_data)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            final_test_auc = test_auc
        # eval_loss_list.append(loss)
        # eval_acc_list.append(test_auc)
    print(f'Epoch: {epoch:03d}, Val: {val_auc:.4f}, '
          f'Test: {test_auc:.4f}')

    y_true = res_cont.flatten()
    # print(true.shape)
    y_pred = res.flatten()
    cm = confusion_matrix(y_true, y_pred)
    # 计算混淆矩阵
    cm = np.zeros((5, 5))
    for i in range(len(y_true)):
        cm[y_true[i]][y_pred[i]] += 1
    print(cm)
    # 绘制混淆矩阵
    heat_map(cm)
    acc = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            if cm[i, :].sum() == 0:
                acc[i] = 0
            else:
                acc[i, j] = cm[i, j] / cm[i, :].sum()
    heat_map_acc(acc)
    return acc

test1(loader1)
plot_acc_loss(eval_loss_list,eval_acc_list)
print(f'Final Test: {final_test_acc:.4f}')

z = model.encode(train_data.x, train_data.edge_index)

final_edge_index = model.decode_all(z)
print(final_edge_index)


time_end = time.time()
print("Elapsed time: %.2f seconds" % (time_end - time_start))

