import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data, download_url, extract_gz


def load_data(data_path):
    '''构建图，得到图上边的连接关系和图上节点的特征'''
    df = pd.read_csv(data_path)
    print(df)
    gene_list = []
    dise_id = df['Gene']
    Gene_id = df['Target_gene']
    for i in dise_id:
        gene_list.append(i)
    for i in Gene_id:
        gene_list.append(i)
    gene_list = set(gene_list)
    gene_list = list(gene_list)
    gene_mapping = {index_id: int(i) + 0 for i, index_id in enumerate(gene_list)}

    print(len(gene_mapping))
    src_nodes = [gene_mapping[index] for index in df['Gene']]
    dst_nodes = [gene_mapping[index] for index in df['Target_gene']]
    edge_index = torch.tensor([src_nodes, dst_nodes])
    rev_edge_index = torch.tensor([dst_nodes, src_nodes])
    gene_feature = pd.read_csv("/home/cuiwentao/cwt/gene/model_multi_task/seq_onehot_feature_pca_300.csv")

    result_gene_feature = pd.read_csv('/home/cuiwentao/cwt/model_final/result_gene_feature_8069.csv')
    # print(result_gene_feature)
    result_gene_feature = result_gene_feature.drop(['gene_name'],axis=1)
    result_gene_feature = result_gene_feature.values
    print(result_gene_feature.shape)
    print(result_gene_feature)
    gene_feature = np.concatenate((gene_feature, result_gene_feature), axis=1)

    data = Data()
    data.num_nodes = len(gene_mapping)
    # data.edge_index = torch.concat([edge_index, rev_edge_index], dim=1)
    data.x = torch.tensor(gene_feature, dtype=torch.float32)
    data.edge_index = torch.concat([edge_index, rev_edge_index], dim=1)
    # data.edge_index = edge_index
    return data, gene_mapping



def get_fre_3(data_c,data_t):
    '''统计每个类别的频数，然后取倒数，返回频数矩阵'''
    change = data_t - data_c
    mask_up = change >= np.log(2)
    # print(mask_up5x)
    # mask_up2x = (change >= np.log(2)) & (change < np.log(5))
    mask_none = (change > -np.log(2)) & (change < np.log(2))
    # mask_down2x = (change <= -np.log(2)) & (change > -np.log(5))
    mask_down = change <= -np.log(2)
    target = pd.DataFrame(np.zeros_like(data_c), dtype=int)

    target[mask_down] = 2
    target[mask_none] = 0
    target[mask_up] = 1
    class_hist, _ = np.histogram(target, bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    label_freq = class_hist / class_hist.sum()
    # print(label_freq)
    target_fre = target
    target_fre.replace(0, 1-label_freq[0],inplace=True)
    target_fre.replace(1, 1-label_freq[1],inplace=True)
    target_fre.replace(2, 1-label_freq[2],inplace=True)
    # print(data.shape)

    return target_fre

def get_fre_5(data_c,data_t):
    '''统计每个类别的频数，然后取倒数，返回频数矩阵'''
    change = data_t - data_c
    # 0/1/2/3/4/5: other/down5x/down2x/none/up2x/up5x
    mask_up5x = change >= np.log(5)
    # print(mask_up5x)
    mask_up2x = (change >= np.log(2)) & (change < np.log(5))
    mask_none = (change > -np.log(2)) & (change < np.log(2))
    mask_down2x = (change <= -np.log(2)) & (change > -np.log(5))
    mask_down5x = change <= -np.log(5)
    target = pd.DataFrame(np.zeros_like(data_c), dtype=int)
    target[mask_down5x] = 4
    target[mask_down2x] = 3
    target[mask_none] = 0
    target[mask_up2x] = 2
    target[mask_up5x] = 1
    class_hist, _ = np.histogram(target, bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    label_freq = class_hist / class_hist.sum()
    target_fre = target
    target_fre.replace(0, 1/(label_freq[0]),inplace=True)
    target_fre.replace(1, 0.1*(label_freq[1]),inplace=True)
    target_fre.replace(2, 0.2*(label_freq[2]),inplace=True)
    target_fre.replace(3, 2*(label_freq[3]),inplace=True)
    target_fre.replace(4, 2*(label_freq[4]),inplace=True)

    return target_fre
