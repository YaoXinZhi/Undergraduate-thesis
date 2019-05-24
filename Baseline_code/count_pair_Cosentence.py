#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:40:59 2019

@author: yaoxinzhi
"""

import re
from functools import reduce
import os

_combinations = lambda x, code=' ': reduce(lambda x, y: [str(i)+code+str(j) for i in x for j in y], x)
    
pair_dic = {}

path = '../result/co_sentence_ParsingTree_sent_2019'

file_list = os.listdir(path)
count = 0
for rf in file_list:
    count += 1
    print (count)
#    print (pair_dic)
    with open('{0}/{1}'.format(path, rf)) as f:
        chem_list = []
        gene_list = []
        for line in f:
            l = line.strip().split('\t')
            try:
                if l[0] == 'sentence':
                    
                    if chem_list != [] and gene_list != []:
                        chem_list = list(set(chem_list))
                        gene_list = list(set(gene_list))
                        for ab in _combinations([gene_list, chem_list]):
    #                        print(ab)
                            gene_id = ab.split()[0]
                            chem_id = ab.split()[1]
                            pair = (chem_id, gene_id)
                            if not pair_dic.get(pair):
                                pair_dic[pair] = 1
                            else:
                                pair_dic[pair] += 1
                    chem_list = []
                    gene_list = []
                if l[0] == 'annotation':
                    ann = l[1].split('|')
                    _id = ann[4]
                    if _id != '':
                        if ann[3] == 'Chemical':
                            if ':' in _id:
                                _id = _id.split(':')[1]
                            chem_list.append(_id)
                        if ann[3] == 'Gene':
                            _id = re.sub(u"\\(.*?\\)", "", _id)
                            if ';' in _id:
                                _id = _id.split(';')[0]
                            gene_list.append(_id)
            except:
                continue

wf = open('../result/chemical_gene_sentence' ,'w')

_key = sorted(pair_dic, key = lambda x:int(pair_dic[x]), reverse=True)
for i in _key:
    wf.write('{0}\t{1}\tGene-Chemical\t{2}\n'.format(i[1], i[0], pair_dic[i]))
wf.close()
