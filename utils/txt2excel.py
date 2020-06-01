# -*- coding: utf-8 -*-

"""
Created on 2020-05-28 11:10
@Author  : Justin Jiang
@Email   : jw_jiang@pku.edu.com
"""

import os
import pandas as pd

def txt2csv(txt_path, csv_path):
    if os.path.exists(txt_path):
        print('-----found file-----')
        data = pd.read_table(txt_path, sep='\t', header=None)
        print(data.shape)
        data.to_csv(csv_path, encoding='utf-8')
        print('-----finished-----')

def read_data_from_excel(csv_path):
    if os.path.exists(csv_path):
        print('-----found csv-----')
        data = pd.read_csv(csv_path)
        print(data)
if __name__ == '__main__':
    train_path = '../data/BoP2017-DBQA.train.txt'
    csv_path = '../data/txt2excel_result.csv'

    txt2csv(train_path, csv_path)
    # read_data_from_excel(csv_path)