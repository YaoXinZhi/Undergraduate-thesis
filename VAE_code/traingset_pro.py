#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 20:16:18 2019

@author: yaoxinzhi
"""
'''
该代码用于将 原始聚类结果_theme 数据集转化为训练节格式
连接通路 并且 分配 label 我们重新整理的标签 7个类
'''

import inspect

class Val:
    # 因为python 没有提供 val.__name__
    def __init__(self, name, num_list, index):
        self.name = name
        self.num_list = num_list
        self.index = index

INHIBITION = Val('INHIBITION', ['3', '6-', '10', '8-', '9-'], 0)
ACTIVATION = Val('ACTIVATION', ['6+', '8+', '9+'], 1)
NEUTRAL_EFFECTS = Val('NEUTRAL_EFFECTS', ['5', '6', '8', '9'], 2)
TREATMENT_RESPONSE = Val('TREATMENT_RESPONSE', ['11a'], 3)
BINDING = Val('BINDING', ['14', '15', '16'], 4)
PRO_ACTS_CHE = Val('PRO_ACTS_CHE', ['19', '20', '21', '11c'], 5)
# 因为是主要的一类错误 在我们的关系中也会有很多 所以考虑保留
UNKNOWN = Val('UNKNOWN', [str(i) for i in range(23,31)], 6)

category_list = [INHIBITION, ACTIVATION, NEUTRAL_EFFECTS, TREATMENT_RESPONSE, BINDING, PRO_ACTS_CHE, UNKNOWN]

def retrieve_name(var):
    # 该函数用于返回变量名
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]


def traingset_pro(rf, out):
    wf = open(out, 'w')
    
    count = 0
    rf = '/home/yaoxinzhi/桌面/project/Graduation/NN/data/chemical-gene-flagship-paths.txt'
    wf1 = open('../data/flg_path_11c', 'w') 
    with open(rf) as f:
        for line in f:
            l = line.strip().split()
            category = l[-1]
            
            # 连接依存路径
            path = l[0].split('|')
            
            if path[-1] == 'start_entity':
                path = list(reversed(l[0].split('|')))
            for i in l[1:-1]:
                tem_list = i.split('|')
                if path[-1] == tem_list[0]:
                    for j in tem_list[1:]:
                        path.append(j)
                elif path[-1] == tem_list[-1]:
                    for j in range(-2, -len(tem_list)-1, -1):
                        path.append(tem_list[j])
#                    tem_list = list(reversed(tem_list))
#                    for j in tem_list[1:]:
#                        path.append(j)
            # 分配标签
            label = ''
            index = ''
            for i in category_list:
                if category in i.num_list:
                    label = i.name
                    index = i.index

            # 保存特定簇的结果 尝试反聚类
            aa = [str(i) for i in range(23, 31)]
            if category not in  aa:
                wf1.write('{0}\t1\n'.format('|'.join(path)))
            
                 
            
            
            
            # group 是文章中分配的小标签
            # index 是我们重新整理的标签
            if label != '':
                count += 1
#                wf.write('{0}\t{1}\t{2}\n'.format('|'.join(path), group, label))
                wf.write('{0}\t{1}\n'.format('|'.join(path), index))
    print ('Done')
    wf1.close()           
    wf.close()
    
if __name__ == '__main__':
    
    rf = '../data/chemical-gene-flagship-paths.txt'
    wf = '../data/flg_path'
    traingset_pro(rf, wf)
    