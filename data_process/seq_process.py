import pandas as pd
import numpy as np
import torch
from collections import Counter
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from Bio import SeqIO
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA



def string_to_array(my_string):
    my_string = my_string.lower()
    # if my_string.find('n'):
    my_string = re.sub('n', 'a',my_string)
    my_array = np.array(list(my_string))
    return my_array

# create a label encoder with 'acgtn' alphabet

def onehot_encode(seq_string):
    label_encoder = LabelEncoder()
    label_encoder.fit(np.array(['a', 'c', 'g', 't']))
    int_encoded = label_encoder.transform(seq_string)
    # print(len(int_encoded))  #2500
    onehot_encoder = OneHotEncoder(sparse=False, dtype=int)
    int_encoded = int_encoded.reshape(len(int_encoded), 1)
    # int_encoded = int_encoded.reshape(-1,1)
    onehot_encoded = onehot_encoder.fit_transform(int_encoded)
    # pca = PCA(n_components=1)
    # pca.fit(onehot_encoded)
    # onehot_encoded = pca.transform(onehot_encoded)
    # onehot_encoded = np.delete(onehot_encoded, -1, 1)
    return onehot_encoded


def dna_sq_process():
    dic_seq_onehot={}
    '''将dna序列变成onehot向量'''
    for sequence in SeqIO.parse("/home/cuiwentao/cwt/gene/mouse_gene_promoter_sequence.fa", "fasta"):
        sequence_id = sequence.id
        sequence_id=sequence_id[0:sequence_id.find(':')]
        # print(sequence_id)
        # print(sequence.seq)
        # print(len(sequence))
        sequence_seq =sequence.seq
        # print(sequence_seq)
        sequence_seq_str = str(sequence_seq)
        seq_test_array = string_to_array(sequence_seq_str)
        # print(seq_test_array)
        onehot_encode1=onehot_encode(seq_test_array)
        # print(onehot_encode1)
        onehot_encode_1dim = [j for i in onehot_encode1 for j in i]
        # print(len(onehot_encode_1dim)) #10000
        # list1 = list(sequence_seq)
        # result = Counter(list1)
        # print(result)
        dic_seq={sequence_id:onehot_encode_1dim}
        dic_seq_onehot.update(dic_seq)
    return dic_seq_onehot
#
# dic_seq_onehot = dna_sq_process()
# seq_onehot_dataframe = pd.DataFrame(dic_seq_onehot)
# print(seq_onehot_dataframe)
# seq_onehot_dataframe.to_csv('seq_onehot_feature_pca.csv',index=False)

def get_seq_feature_choosecol():
    '''通过8069个基因的名字选出seq特征中对应的列'''
    data=pd.read_csv('/home/cuiwentao/cwt/gene/tf_ko108.csv')
    print(data)
    gene_name623 = list(data['index'])

    pca_feature = pd.read_csv('/home/cuiwentao/cwt/gene/model_multi_task/seq_onehot_feature_pca.csv')
    # print(pca_feature)
    pca_feature = pca_feature[gene_name623]#选取部分列
    print(pca_feature)
    pca_feature.to_csv('seq_feature8069_10000.csv',index=False)

# get_seq_feature_choosecol()


def pca_seq():
    '''pca降维：将seq10000维特征降维到指定维数'''
    data_onehot = pd.read_csv('//home/cuiwentao/cwt/gene/model_multi_task/seq_feature8069_10000.csv')
    # for i in data_onehot.columns:
    # print(data_onehot[i])
    # col = np.array(list(data_onehot[i])).reshape(-1,1)
    col=np.array(data_onehot).T
    print(col)
    print(col.shape)
    pca = PCA(n_components=300)
    col = pca.fit_transform(col)
    # dic_seq = {i: col}
    # dic_seq_onehot.update(dic_seq)
    seq_dataframe = pd.DataFrame(col)
    print(seq_dataframe)
    seq_dataframe.to_csv('seq_onehot_feature_pca_300.csv',index=False)
    print(col)
    print(col.shape)

pca_seq()
data_onehot = pd.read_csv('/home/cuiwentao/cwt/gene/pac_feature8069_10000.csv')
print(data_onehot)



def feature_ko_process():
    '''将要敲出的基因变成onehot向量'''
    gene_data= pd.read_csv(r"D:\study\code\mus_kc_7.csv")
    gene_data =gene_data.iloc[:,986:]
    # print(gene_data['Tet1_Tet2_Tet3'])

    columns_name = gene_data.columns
    columns_name = list(columns_name)
    print(columns_name)
    # print(columns_name.index('Vi'))
    name_gene = pd.read_csv(r"D:\study\code\name_gene1.csv")
    gene_name = list(name_gene['name'])
    print(gene_name)
    # line[0:line.rfind("] : ")]
    index_gene_name=[]
    feature_ko = []
    for i in columns_name:
        i=i.replace('_','.')
        if i.find('.'):
            i = i[0:i.find('.')]
            gene_index = gene_name.index(i)
            index_gene_name.append(gene_index)
            j = 0
            e = [0.0] * 27454
            # print(gene_index)
            e[gene_index] = 1
            feature_ko.append(e)
            j += 1
        else:
            gene_index = gene_name.index(i)
            index_gene_name.append(gene_index)
            j = 0
            e = [0.0] * 27454
            # print(gene_index)
            e[gene_index] = 1
            feature_ko.append(e)
            j += 1
    print(len(feature_ko[0]))
    feature_ko = pd.DataFrame(feature_ko).T
    print(feature_ko)
    feature_ko.to_csv('feature_ko.csv',index=False)


# feature_ko_process()