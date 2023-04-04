import os
import pandas as pd

#将数据分组，每组内分为敲出前和敲除后，在分别取平均数。每个原始文件内重复的基因表达值相加
#取出小鼠基因
def data_merge():
    Path="/home/cuiwentao/lz/grn/data/mouse/mus" # 文件夹路径
    fileList=os.listdir(Path)
    # df = pd.read_table(filePath + '\\' + 'SRR2007105' + '_expression.txt')
    # print(df)

    srr = pd.read_excel("/home/cuiwentao/cwt/3328/srr.xlsx")
    print(srr)
    df_k_zong = pd.DataFrame(columns=['Gene Name'])#敲除后
    df_c_zong = pd.DataFrame(columns=['Gene Name'])
    print(df_k_zong)
    srr_group = srr.groupby('group')
    p = 1
    l = 1
    w=1
    q=1
    for i,j in srr_group:
        # print(j)
        if len(list(set(list(j['ko'])))) > 1:
            group_ko_group = j.groupby('ko')
            for k,h in group_ko_group:
                print('group_ko_group:{}'.format(h))
                srr_name_group = list(h['SRR'])
                # print(srr_name_group)
                count=len(srr_name_group)
                ko_type = list(h['ko'])
                print('ko_type[0]:{}'.format(ko_type[0]))
                df_k = pd.DataFrame(columns=['Gene Name'])  # 敲除后
                df_c = pd.DataFrame(columns=['Gene Name'])
                if ko_type[0]=='wild type':
                    print('..............')

                    for name in srr_name_group:
                        # print(name)
                        if (os.path.exists(Path+'/'+str(name)+'_expression.txt')):
                            df = pd.read_table(Path+'/'+str(name)+'_expression.txt')
                            print(df)
                            df = df.loc[:, ['Gene Name', 'FPKM']]
                            df = df.groupby('Gene Name').sum()  # 将重复行加和
                            print(str(ko_type[0]) + '_' + str(p))
                            df.columns = [str(ko_type[0]) + '_' + str(p)]

                            df_c = pd.merge(df_c, df, how='outer', on='Gene Name').fillna(0)  #
                            p=int(p)
                            p=p+1
                            print(df_c)
                    if (os.path.exists(Path + '/' + str(name) + '_expression.txt')):
                        # df_non_name_c = df_k.iloc[:, 1]
                        df_c['mean'] = df_c.mean(axis=1)
                        print(df_c)
                        df_c = df_c[['Gene Name', 'mean']]
                        print(df_c)
                        df_c.columns = ['Gene Name', str(ko_type[0]) + '_' + str(q)]
                        df_c_zong = pd.merge(df_c_zong, df_c, how='outer', on='Gene Name').fillna(0)  #
                        print(df_c_zong)
                        q=q+1

                else:

                    for name in srr_name_group:
                        # print(name)

                        if (os.path.exists(Path + '/' + str(name) + '_expression.txt')):
                            print(ko_type[0])
                            print('111111111111111')
                            df1 = pd.read_table(Path + '/' + str(name) + '_expression.txt')
                            print(df1)
                            df1 = df1.loc[:, ['Gene Name', 'FPKM']]
                            df1 = df1.groupby('Gene Name').sum()  # 将重复行加和

                            df1.columns=[str(ko_type[0])+'_'+str(l)]

                            df_k = pd.merge(df_k, df1, how='outer', on='Gene Name').fillna(0)  #
                            l=l+1
                            print(df_k)
                    if (os.path.exists(Path + '/' + str(name) + '_expression.txt')):
                        # df_non_name = df_k.iloc[:,1]
                        df_k['mean'] = df_k.mean(axis=1)
                        print(df_k)
                        df_q=df_k[['Gene Name','mean']]
                        print(df_q)
                        print(w)
                        df_q.columns = ['Gene Name', str(ko_type[0]) + '_' + str(w)]
                        print(df_q)
                        print(df_k_zong)
                        df_k_zong = pd.merge(df_k_zong, df_q, how='outer', on='Gene Name').fillna(0)  #

                        print(df_k_zong)
                        w=w+1
    print(df_c_zong)
    print(df_k_zong)
    df_c_zong.to_csv('a1df_c1221_bad.csv',index=False)
    df_k_zong.to_csv('a1df_t1221_bad.csv',index=False)

def data_choose():
    data_c = pd.read_csv('/home/cuiwentao/cwt/3328/df_c1221_bad_new.csv')
    data_t = pd.read_csv('/home/cuiwentao/cwt/3328/df_t1221_bad_new.csv')
    data_c.set_index('Gene Name', drop=True, append=False, inplace=True)
    data_t.set_index('Gene Name', drop=True, append=False, inplace=True)
    gene_name = pd.read_csv('/home/cuiwentao/cwt/name_gene1.csv')
    name = list(gene_name['name'])
    data_c=data_c.loc[name]
    print(data_c)
    data_t = data_t.loc[name]
    print(data_t)
    data_c.to_csv('adf_c1221_new_bad.csv', index=False)
    data_t.to_csv('adf_t1221_new_bad.csv', index=False)
# data_choose()

def count_zero(addr):
    data = pd.read_csv(addr)
    count0 = (data == 0).astype(int).sum(axis=0)
    print(count0)
    print(type(count0))

    pd.DataFrame(count0).to_csv('aa_t.csv', index=False)

    data_t = data.T
    count0_gene = (data_t == 0).astype(int).sum(axis=0)
    print(count0_gene)
    pd.DataFrame(count0_gene).to_csv('aa_gene_t.csv', index=False)

count_zero('/home/cuiwentao/cwt/3328/ana1221/adf_t1221_new.csv')
# count_zero('/home/cuiwentao/cwt/3328/ana1221/adf_c1221_new.csv')
# count_zero('/home/cuiwentao/cwt/3328/ana1221/adf_c1221_new_ser.csv')
# count_zero('/home/cuiwentao/cwt/3328/ana1221/adf_t1221_new_ser.csv')
