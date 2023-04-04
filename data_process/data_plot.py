import pandas as pd
from get_acc import get_acc
from get_acc import crosstable
from get_acc import tag_test
import numpy as np
# from plot_matrix import plt_matrix
# from plot_matrix import plt_matrix1
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

def column_sum(data):
    count0 = (data == 0).astype(int).sum(axis=0)
    count1 = (data == 1).astype(int).sum(axis=0)
    count2 = (data == 2).astype(int).sum(axis=0)
    count3 = (data == 3).astype(int).sum(axis=0)
    count4 = (data == 4).astype(int).sum(axis=0)
    list_sum=[]
    list_sum.append(count0)
    list_sum.append(count1)
    list_sum.append(count2)
    list_sum.append(count3)
    list_sum.append(count4)
    data_sum = pd.DataFrame(list_sum)
    # print(data_sum)
    return data_sum


# data = pd.read_csv('/home/cuiwentao/cwt/mus_kc_7.csv')
# print(data)
# data=np.log(data+1)

data_t_y = pd.read_csv('/home/cuiwentao/cwt/3328/ana1221/adf_t1221_new_bad.csv')
data_c= pd.read_csv('/home/cuiwentao/cwt/3328/ana1221/adf_c1221_new_bad.csv')
data_t =np.log(data_t_y+1)
data_c =np.log(data_c+1)
data_c =data_c.values
data_t = data_t.values

tag,change = tag_test(data_c,data_t)
tag=pd.DataFrame(tag)
print(tag)


column_name_list_add  =list(data_t_y.columns)
# print(columns_name)
columns_name =column_name_list_add
col_name = []
for i in columns_name:
    i = i.replace('/', '_')
    if i.find('_') == -1:
        col_name.append(i)
    else:
        i = i[0:i.find('_')]
        col_name.append(i)
print(col_name)
column_name_list_add = col_name
col_name_1 = {'name':col_name}
column_name = pd.DataFrame(col_name_1)

# column_name_list_add = column_name
tag.columns =column_name_list_add
print(tag)
change_data = pd.DataFrame(change)
change_data.columns =column_name_list_add
index=[]
for i in range(column_name.shape[0]):
    index.append(i)
column_name['index']=index
print(column_name)

column_name_group = column_name.groupby(column_name['name'])



# index_start=0
# index_end=1
for j,i in column_name_group:
    print(i)
    i_shape=i.shape[0]
    if i.shape[0] > 1:
        index_name = i.iloc[0:1,0:1].values
        # index_end = int(i.iloc[-1:, 1:2].values)
        # print(index_start)
        # print(index_end)
        index_name=str(index_name[0][0])
        # print(index_name)
        print(tag)
        # # print(tag['Adar'])

        tag_choose = tag[index_name]

        print(tag)

        change_data_choose = change_data[index_name]#得到源数据用于分析

        # column_name = column_name.iloc[index_start:index_end,0:1]#根据分组的索引到源数据中取名字，用于绘图的图片名
        column_name_list = list(i['name'])
        # print(column_name_list)
        # print('1111111111111111')

        tag1 = tag_choose.values
        data_c123=column_sum(tag1)
        data_c123.columns=column_name_list
        data_c123=data_c123.T
        data_c123.columns=['down5x','down2x','none','up2x','up5x']
        print(data_c123)
        # data_c123.to_csv('{}_count_fre.csv'.format(column_name_list[0]))

        #条形图
        data_c123.plot.bar()
        plt.tight_layout()
        plt.show()
        plt.savefig("{}_bar.jpg".format(index_name))
        plt.close()
        #箱型图
        change_data_choose.plot(kind='box')
        # data.plot(kind='hist')
        # plt.xlabel("分组")
        plt.title("{}".format(index_name))
        # plt.ylabel(r"频率/%")
        plt.show()
        plt.savefig("{}_box.jpg".format(index_name))
        plt.close()

        #将数据统计情况写进csv
        data_describe = change_data_choose.describe()
        data_describe.to_csv('{}_describe.csv'.format(index_name))

        tag_data = tag_choose.T

        print(column_name_list)
        tag_data['columns_name'] = column_name_list

        data_group = tag_data.groupby(tag_data['columns_name'])
        print(data_group)

        for j,i in data_group:
            # print(i)
            # print(type(i)) #dataframe
            i = i.iloc[:,:-1] #去掉添加的列名
            count_down5x = 0
            count_down2x = 0
            count_none = 0
            count_up2x = 0
            count_up5x = 0
            count_zong = 0
            data_group_each=i
            if data_group_each.shape[0]>1:
                # print(c)
                data_group_each=data_group_each.T #转置后行是基因，列是实验
                # print(c.count(0))
                for k in range(data_group_each.shape[0]):
                    #查看每行值是否相同，统计值相同的数量
                    da = data_group_each.iloc[k:k+1,:]
                    # print(da)
                    # print(da.iat[k])
                    da_sum = da.sum(axis=1)
                    # print(da_sum)
                    value_first = da.iloc[0:1, 0:1].values
                    if da_sum.values == ((value_first)*da.shape[1]):
                        count_zong += 1
                        if value_first == 0:
                            count_down5x += 1
                        elif value_first == 1:
                            count_down2x += 1
                        elif value_first == 2:
                            count_none += 1
                        elif value_first == 3:
                            count_up2x += 1
                        elif value_first == 4:
                            count_up5x += 1
            dic_count_repeat = {'count_down5x': count_down5x, 'count_down2x': count_down2x, 'count_none': count_none,
                                'count_up2x': count_up2x, 'count_up5x': count_up5x, 'count_zong': count_zong}
            list_count_repeat = (count_down5x, count_down2x, count_none, count_up2x,  count_up5x)
            print(dic_count_repeat)

            data_c123.loc['overlap'] = list_count_repeat
            data_c123.to_csv('{}_count_fre.csv'.format(index_name))

