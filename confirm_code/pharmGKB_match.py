#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 15:05:43 2019

@author: yaoxinzhi
"""

import argparse

def PharmGKB_match(ent_file, percentage):
    
    # 读取已知关系对
    pharm_list = []
    pharm_relationships = '../database/PharmGKB/relationships/relationships.tsv'
    with open(pharm_relationships) as f:
        for line in f:
            l = line.split('\t')
            ent1_type = l[2]
            ent2_type = l[5]
            gene_name = chem_name = ''
            if ent1_type == 'Chemical' and ent2_type == 'Gene':
                gene_name = l[4]
                chem_name = l[1]
            if ent1_type == 'Gene' and ent2_type == 'Chemical':
                gene_name = l[1]
                chem_name = l[4]
            gene_name = ''.join(e for e in gene_name if e.isalnum()).lower()
            chem_name = ''.join(e for e in chem_name if e.isalnum()).lower()
            if gene_name != '' and chem_name != '':
                relation_pair = (chem_name, gene_name)
                if relation_pair not in pharm_list:
                    pharm_list.append(relation_pair)
    print (len(pharm_list))
    
#     读取实体对列表                    
    ent_pair = []
    with open(ent_file) as f:
        for line in f:
            l = line.split('\t')
            chem_name = l[1]
            gene_name = l[2]
            chem_name = ''.join(e for e in chem_name if e.isalnum()).lower()
            gene_name = ''.join(e for e in gene_name if e.isalnum()).lower()
            if chem_name != 'None' and gene_name != 'None':
                ent_pair.append((chem_name, gene_name))
#    print (ent_pair)
#            
#    # 查看前 百分之x 有多少在PharmGKB 已知关系对中
#    wf = open(out, 'w')
    x = float(percentage)
    count = 0
    
    for i in ent_pair[:int(x * len(ent_pair))]:
        if i in pharm_list:
            count += 1
#            wf.write('{0}\t{1}\tKnown\n'.format(i[0], i[1]))
#        else:
#            wf.write('{0}\t{1}\tUnknown\n'.format(i[0], i[1]))
    print ('在前 {0}% ({1}个)关系对中, 有 {2} 在PharmGKB 已知关系对中'.format(x*100, int(len(ent_pair)*x), count))

if __name__ == '__main__':
    
    parse = argparse.ArgumentParser()
    parse.add_argument('-r', '--rfile', help='Source file name',
                   dest='rf', required=True)
#    parse.add_argument('-w', '--wf', help='wf file name',
#                        dest='wf', required=True)
    parse.add_argument('-p', '--per', help='percentage',
                        dest='per', required=True)
    args = parse.parse_args()
    
#    ent_file = '../result/ent_pair_same'
#    wf = open('../result/known_pair', 'w')
    
    PharmGKB_match(args.rf, args.per)