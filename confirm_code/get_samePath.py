#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 14:47:56 2019

@author: yaoxinzhi
"""

'''
该代码用于比较 我们抽提的 全pubmed路径和 700条paper中的旗舰路径有多少一样
占700条中的多少 （覆盖度）
'''

def SamePath(pub_path, flg_path, out):
    
    wf = open(out, 'w')
    wf_over = open('{0}_overpath'.format(out), 'w')
    
    
    flg_list = []
    with open(flg_path) as f:
        for line in f:
            l = line.strip().split()
            flg_list.append((l[0], l[-1]))
    
    over_path = {}
    print ('extracting same paht \n')
    count = 0
    with open(pub_path) as f:
        for line in f:
            l = line.strip().split('\t')
            path_list = l[0].split('|')
            if len(path_list) in range(1, 11):
                for i in flg_list:
                    if l[0] == i[0]:
                        count += 1
                        
                        if l[0] not in over_path.keys():
                            over_path[l[0]] = 1
                        else:
                            over_path[l[0]] += 1
                        
                        for j in l:
                            wf.write(j+'\t')
                        wf.write('\t{0}\n'.format(i[1]))
                        
    key_sorted = sorted(over_path, key = lambda x:over_path[x], reverse=True)
    for i in key_sorted:
        wf_over.write('{0}\t{1}\n'.format(i, over_path[i]))
    wf_over.close()
    wf.close()
    print ('Done')

if __name__ == '__main__':
    
    pub_path = '../data/chemical_gene_DepPath'
    flg_path = '../data/flg_path'
    wf = '../result/same_path'
    
    SamePath(pub_path, flg_path, wf)