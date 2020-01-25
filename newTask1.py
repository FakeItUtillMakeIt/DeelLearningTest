
#import os
import numpy as np
from sklearn.svm import SVC
#from sklearn.tree import DecisionTreeClassifier
from xlrd import open_workbook
import sklearn.preprocessing as pre_processing
from sklearn.preprocessing import OneHotEncoder


filename=r"C:\Users\1\Desktop\全部隧道洞身电阻率.xls"
excel=open_workbook(filename)
excel_sheet=excel.sheet_by_index(0)

print(excel_sheet.name,excel_sheet.nrows,excel_sheet.ncols)
#获取列数，
col_name=[]
for col_index in range(excel_sheet.ncols):
    col_name.append(excel_sheet.col(col_index,0,1)[0].value)

#获取对应特征索引，并根据列名寻找相关特征,标签
def get_index(col_names,index_name):
    index=[]
    for name in index_name:
        index.append(col_name.index(name))
    return index

col_index=get_index(col_name,['电阻率平均值','岩性物理分类','风化物理分类','围岩等级'])


def sureCate(column_data):
    cate_num=[]
    for data in column_data:
        if data not in cate_num:
            cate_num.append(data)
        else:
            pass

    return cate_num

#获取训练集及测试集
def get_dataset(excel_sheets,column_index):
    data_tup=[]
    for column in column_index:
        feature=excel_sheet.col_values(column,0,1)
        column_data = excel_sheet.col_values(column, 1)

        if feature[0]=='岩性物理分类' or feature[0]=='风化物理分类':
            #column_data=character_to_num(column_data)
            #对表里面的非数值化数据进行数值化，对风化物理分类和岩性物理分类及围岩等级进行数值化,独热编码
            pre_lab=pre_processing.LabelEncoder()
            column_data=pre_lab.fit_transform(column_data)
            print(column_data)
            # 确定每列类别个数
            cate = sureCate(column_data)
            print((cate))
            #独热编码
            pre_one=OneHotEncoder()
            cate=np.asarray(cate)
            cate=cate.reshape(cate.shape[0],1)
            fit_fun=pre_one.fit(cate)
            column_data=np.asarray(column_data)
            column_data=column_data.reshape(column_data.shape[0],1)
            column_data=fit_fun.transform(column_data).toarray()
            print(column_data)
        elif feature[0]=='围岩等级':
            pre_lab = pre_processing.LabelEncoder()
            column_data = pre_lab.fit_transform(column_data)

        data_tup.append(column_data)
    dataset=np.column_stack(data_tup)
    return dataset

dataset=get_dataset(excel_sheet,col_index)

#SVM
svm_model=SVC()
svm_model.fit(dataset[:,:-1],dataset[:,-1])



