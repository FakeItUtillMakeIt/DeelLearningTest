
#import os
import numpy as np
from sklearn.svm import SVC
#from sklearn.tree import DecisionTreeClassifier
from xlrd import open_workbook
import sklearn.preprocessing as pre_processing


filename=r"C:\Users\1\Desktop\全部隧道洞身电阻率.xls"
excel=open_workbook(filename)
excel_sheet=excel.sheet_by_index(0)

print(excel_sheet.name,excel_sheet.nrows,excel_sheet.ncols)
#获取列数，并根据列名寻找相关特征,标签
col_name=[]
for col_index in range(excel_sheet.ncols):
    col_name.append(excel_sheet.col(col_index,0,1)[0].value)

#获取索引
def get_index(col_names,index_name):
    index=[]
    for name in index_name:
        index.append(col_name.index(name))
    return index

col_index=get_index(col_name,['电阻率平均值','岩性物理分类','风化物理分类','围岩等级'])

#变更非数字类型为数字类型
def character_to_num(column_dataset):
    dataset=[]
    for data in column_dataset:
        if data == 'Ⅱ' or data == 'Ⅱ':
            dataset.append(2)
        elif data == 'Ⅲ' or data == 'III':
            dataset.append(3)
        elif data == 'Ⅳ' or data =='IV':
            dataset.append(4)
        elif data == 'Ⅴ' or data == 'ⅴ' or data == 'V':
            dataset.append(5)
        elif data == 'Ⅵ' or data == 'VI':
            dataset.append(6)
        else:
            dataset.append(0)

    return dataset

def lithopic_to_num(column_dataset):
    dataset=[]
    for data in column_dataset:
        if data == 'Ⅱ' or data == 'Ⅱ':
            dataset.append(2)
        elif data == 'Ⅲ' or data == 'III':
            dataset.append(3)
        elif data == 'Ⅳ' or data =='IV':
            dataset.append(4)
        elif data == 'Ⅴ' or data == 'ⅴ' or data == 'V':
            dataset.append(5)
        elif data == 'Ⅵ' or data == 'VI':
            dataset.append(6)
        else:
            dataset.append(0)

    return dataset


#获取训练集及测试集
'''
def get_dataset(excel_sheets,column_index):
    data_tup=[]
    for column in column_index:
        data_tup.append(excel_sheets.col_values(column,1))
    dataset=np.column_stack(data_tup)
    return dataset
'''
def get_dataset(excel_sheets,column_index):
    data_tup=[]
    for column in column_index:
        feature=excel_sheet.col_values(column,0,1)
        column_data = excel_sheet.col_values(column, 1)
        if feature[0]=='围岩等级' or feature[0]=='岩性物理分类' or feature[0]=='风化物理分类':
            #column_data=character_to_num(column_data)
            #对表里面的非数值化数据进行数值化，对风化物理分类和岩性物理分类及围岩等级进行数值化
            pre_lab=pre_processing.LabelEncoder()
            column_data=pre_lab.fit_transform(column_data)
        data_tup.append(column_data)
    dataset=np.column_stack(data_tup)
    return dataset

dataset=get_dataset(excel_sheet,col_index)

'''
tree_model=DecisionTreeClassifier()
tree_model.fit(dataset[:,:-1],dataset[:,-1])

'''

svm_model=SVC()
svm_model.fit(dataset[:,:-1],dataset[:,-1])