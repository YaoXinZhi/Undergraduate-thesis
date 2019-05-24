#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 15:10:55 2019

@author: yaoxinzhi
"""

'''
该文件用于尝试匹配TTD数据库中已知的 chemical-target 关系对
'''

import argparse

def ttd_match(ent_file, percentage):
    
    # 读取 uniprot id 2 entrze id
    with open('/home/yaoxinzhi/桌面/project/Graduation/confirm/database/TTD/uniprot_mapping.tab') as f:
        uniprot_mapping = {}
        for line in f:
            l = line.strip().split('\t')
            try:
                uniprot_mapping[l[0]] = l[1]
            except:
                continue
    
#    # get gene symbol from ../database/GO/goa_human.gaf 只匹配了人的gene
#    with open('/home/yaoxinzhi/桌面/project/Graduation/confirm/database/GO/goa_human.gaf') as f:
#        uniprot2symbol = {}
#        for line in f:
#            l = line.strip().split('\t')
#            try:
#                if l[0] == 'UniProtKB':
#                    uniprot2symbol[l[1]] = l[2]
#            except:
#                continue
#    
    # read TTD from the '../database/TTD/P1-01-TTD_download.txt
    Metabolism = ['Immunomodulator', 'Inducer', 'Immunomodulator (Immunostimulant)', 'Regulator', 'Cofactor', 'Stimulator']
    Transport = ['Binder', 'Breaker', 'Stablizer', 'Binder (minor groove binder)', 'Intercalator', 'Modulator', 'Modulator (allosteric modulator)', 'Opener']
    Inhibition = ['Inhibitor', 'Inhibitor (gating inhibitor)']
    Agonism = ['Activator', 'Agonist', 'Enhancer', 'Regulator (upregulator)']
    Antagonism = ['Antagonist', 'Blocker', 'Suppressor', 'Blocker (channel blocker)']
    ttd_download = '/home/yaoxinzhi/桌面/project/Graduation/confirm/database/TTD/P1-01-TTD_download.txt'
#    count = [0, 0, 0, 0, 0, 0]
    with open(ttd_download) as f:
        ttd_relationship = {}
        for line in f:
            l = line.strip().split('\t')
            try:
                if l[1] == 'UniProt ID':
                    
                    _id = l[2]
                    if len(_id) > 6:
                        _id = _id[:6]
                        
                    if uniprot_mapping.get(_id):
                        Entrze_id = uniprot_mapping[_id]
                        if not ttd_relationship.get(Entrze_id):    
                            ttd_relationship[Entrze_id] = {'Drug': [], 'Metabolism' : [], 'Transport': [], 'Inhibition' : [], 'Agonism' : [], 'Antagonism' : []}
                    else:
                        Entrze_id = ''
                        
                if l[1] == 'Drug(s)' and Entrze_id != '':
#                    count[0] += 1
                    chemical_name = ''.join(e for e in l[2] if e.isalnum()).lower()
                    ttd_relationship[Entrze_id]['Drug'].append(chemical_name)
                if l[1] in Metabolism and Entrze_id != '':
#                    count[1] += 1
                    chemical_name = ''.join(e for e in l[2] if e.isalnum()).lower()
                    ttd_relationship[Entrze_id]['Metabolism'].append(chemical_name)
                if l[1] in Transport and Entrze_id != '':
#                    count[2] += 1
                    chemical_name = ''.join(e for e in l[2] if e.isalnum()).lower()
                    ttd_relationship[Entrze_id]['Transport'].append(chemical_name)
                if l[1] in Inhibition and Entrze_id != '':
#                    count[3] += 1
                    chemical_name = ''.join(e for e in l[2] if e.isalnum()).lower()
                    ttd_relationship[Entrze_id]['Inhibition'].append(chemical_name)
                if l[1] in Agonism and Entrze_id != '':
#                    count[4] += 1
                    chemical_name = ''.join(e for e in l[2] if e.isalnum()).lower()
                    ttd_relationship[Entrze_id]['Agonism'].append(chemical_name)
                if l[1] in Antagonism and Entrze_id != '':
#                    count[5] += 1
                    chemical_name = ''.join(e for e in l[2] if e.isalnum()).lower()
                    ttd_relationship[Entrze_id]['Antagonism'].append(chemical_name)
            except:
                continue

    # 删除重复
    for i in ttd_relationship.keys():
        for j in ttd_relationship[i].keys():
            ttd_relationship[i][j] = list(set(ttd_relationship[i][j]))
            
    # 读取我们的结果
    ent_file = '/home/yaoxinzhi/桌面/project/Graduation/confirm/result/ent_vae'
    with open(ent_file) as f:
        pairs_list = []
        for line in f:
            l = line.strip().split('\t')
            entrze_id = eval(l[0])[1]
            chemical_name = ''.join(e for e in l[1] if e.isalnum()).lower()
            pairs_list.append((chemical_name, entrze_id))

    # 储存gene_list
    gene_list = []
    for i in pairs_list:
        gene_list.append(i[1])
    gene_list = list(set(gene_list))

    # 只保留有记载的gene
    ttd_relationship_match = {}
    for i in ttd_relationship.keys():
        if i in gene_list:
            ttd_relationship_match[i] = ttd_relationship[i]
            
    # 计算每种关系计数
    count = {'Drug': 0, 'Metabolism' : 0, 'Transport': 0, 'Inhibition' : 0, 'Agonism' : 0, 'Antagonism' : 0}
    for i in ttd_relationship_match.keys():
        for j in ttd_relationship_match[i].keys():
            for k in ttd_relationship_match[i][j]:
                count[j] += 1
    
    # 读取同义词表 ../database/TTD/P1-03-Drug_Synonyms.txt
    synoyms_list = []
    with open('/home/yaoxinzhi/桌面/project/Graduation/confirm/database/TTD/P1-03-Drug_Synonyms.txt') as f:
        for line in f:
            temp_list = []
            l = line.strip().split('\t')
#            try:
            for i in l[1:]:
                if ';' in i:
                    for j in i.split(';'):
                        chemical_name = ''.join(e for e in j if e.isalnum()).lower()
                        temp_list.append(chemical_name)
                if ',' in i:
                    for j in i.split(','):
                        chemical_name = ''.join(e for e in j if e.isalnum()).lower()
                        temp_list.append(chemical_name)
                else:
                    chemical_name = ''.join(e for e in i if e.isalnum()).lower()
                    temp_list.append(chemical_name)
            temp_list = list(set(temp_list))
            synoyms_list.append(temp_list)

    # 添加同义词
    aa = 0
    for i in ttd_relationship.keys():
        aa += 1
        if aa% 500 == 0:
            print (aa)
        for j in ttd_relationship[i].keys():

#            print (ttd_relationship[i][j])
#            print (ttd_relationship[i][j])
            temp_list = ttd_relationship[i][j]
            syno_list = []
            for k, v in enumerate(temp_list):
                for sy in synoyms_list:
                    if v in set(sy):
                        syno_list += sy
            syno_list += temp_list
            
            ttd_relationship[i][j] = syno_list
            
#            print (syno_list)


##     match
    x = float(percentage)
    x = 0.1
    count_result = [0, 0, 0, 0, 0, 0]
    for pair in pairs_list[:int(x * len(pairs_list))]:
        if ttd_relationship.get(pair[1]):
            if pair[0] in ttd_relationship[pair[1]]['Drug']:
                count_result[0] += 1
            if pair[0] in ttd_relationship[pair[1]]['Metabolism']:
                count_result[1] += 1
            if pair[0] in ttd_relationship[pair[1]]['Transport']:
                count_result[2] += 1
            if pair[0] in ttd_relationship[pair[1]]['Inhibition']:
                count_result[3] += 1
            if pair[0] in ttd_relationship[pair[1]]['Agonism']:
                count_result[4] += 1
            if pair[0] in ttd_relationship[pair[1]]['Antagonism']:
                count_result[5] += 1
    print ('在前 {0}% ({1}个)关系对中,\n有 {2} 在TTD 已知关系对\n[Drug, Metabolism, Transport, Ingibition, Agonism, Antagonism]\n共 {3}'.format(x*100, int(len(pairs_list)*x), count_result, count))
        
if __name__ == '__main__':
    
    parse = argparse.ArgumentParser()
    parse.add_argument('-r', '--rfile', help='Source file name',
                   dest='rf', required=True)
    parse.add_argument('-p', '--per', help='percentage',
                        dest='per', required=True)
    args = parse.parse_args()
    
#    ent_file = '../result/ent_pair_same'
#    wf = open('../result/known_pair', 'w')
    
    ttd_match(args.rf, args.per)