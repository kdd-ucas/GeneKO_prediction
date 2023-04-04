import numpy as np
import pandas as pd

import seaborn as sns
from collections import Counter

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
def tag_test(datac,datat,count_gene):

    datac=datac.cpu().numpy()
    datat = datat.cpu().numpy()
    print(datac.shape)
    datac = datac[:,0:count_gene]
    print(datac.shape)
    change = datat - datac
    # print(change)
    mask_up5x = change >= np.log(5)
    # print(mask_up5x)
    mask_up2x = (change >= np.log(2)) & (change < np.log(5))
    mask_none = (change > -np.log(2)) & (change < np.log(2))
    mask_down2x = (change <= -np.log(2)) & (change > -np.log(5))
    mask_down5x = change <= -np.log(5)
    # target = pd.DataFrame(np.zeros_like(data_c), index=data_c.index, columns=data_c.columns, dtype=int)
    target = pd.DataFrame(np.zeros_like(datac),dtype=int)
    target[mask_down5x] = 4
    target[mask_down2x] = 3
    target[mask_none] = 0
    target[mask_up2x] = 2
    target[mask_up5x] = 1
    # print(target)
    target = target.values
    print(target)
    print(target.shape)

    return target

def tag_test1(datac,datat,count_gene):
    datac=datac.cpu().numpy()
    datat = datat.cpu().numpy()
    print(datac.shape)
    datac = datac[:,0:count_gene]
    print(datac.shape)
    change = datat - datac
    # print(change)
    mask_up = change >= np.log(5)
    # print(mask_up5x)
    # mask_up = (change >= np.log(2)) & (change < np.log(5))
    mask_none = (change > -np.log(5)) & (change < np.log(5))
    # mask_down2x = (change <= -np.log(2)) & (change > -np.log(5))
    mask_down = change <= -np.log(5)
    # target = pd.DataFrame(np.zeros_like(data_c), index=data_c.index, columns=data_c.columns, dtype=int)
    target = pd.DataFrame(np.zeros_like(datac),dtype=int)
    target[mask_down] = 2
    target[mask_none] = 0
    target[mask_up] = 1
    # print(target)
    target = target.values
    print(target)
    print(target.shape)

    return target


def heat_map(cm):
    # 绘制 heatmap
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Oranges)

    # 设置颜色条
    plt.colorbar()

    # 设置坐标轴标签
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')

    # 设置坐标轴刻度
    tick_marks = np.arange(5)
    plt.xticks(tick_marks, ['down_5x', 'down_2x', 'none', 'up_2x', 'up_5x'])
    plt.yticks(tick_marks, ['down_5x', 'down_2x', 'none', 'up_2x', 'up_5x'])

    # 在格子中显示数字
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j],'.0f'),
                 ha='center', va='center',
                 color='black' if cm[i, j] > thresh else 'black')
    plt.savefig('count.png')

    plt.show()
    plt.close()

def heat_map_acc(acc):
    # 绘制 heatmap
    # cmap = plt.cm.get_cmap('Blues')
    plt.imshow(acc, interpolation='nearest', cmap=plt.cm.Oranges)


    # 设置颜色条
    plt.colorbar()

    # 设置坐标轴标签
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')

    # 设置坐标轴刻度
    tick_marks = np.arange(5)
    plt.xticks(tick_marks, ['down_5x', 'down_2x', 'none', 'up_2x', 'up_5x'])
    plt.yticks(tick_marks,['down_5x', 'down_2x', 'none', 'up_2x', 'up_5x'])

    # 在格子中显示数字
    for i, j in np.ndindex(acc.shape):
        plt.text(j, i, format(acc[i, j], '.2f'),
                 ha='center', va='center',
                 color='black')
    plt.savefig('acc.png')
    plt.show()
    plt.close()