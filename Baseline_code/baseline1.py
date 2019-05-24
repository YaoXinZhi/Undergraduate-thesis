#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 22:59:31 2019

@author: yaoxinzhi
"""

'''
baseline 1 
提取基于文章共显的 gene-chemical 实体对
'''

import argparse
import os
from itertools import combinations
#from itertools import product
from functools import reduce
import xml.etree.ElementTree as ET
import re

def Result_pair(rf, out, node_a, node_b, mesh, ebi, supp):
    single_pair = open('{0}/{1}_spair'.format(out, rf.split('/')[-1]), 'w')

    annotation = {}
    tem_a = []
    tem_b = []
    pair_aa = []
    pair_bb = []
    all_pair = {}
    count_a = {}
    count_b = {}
    
    
    # 读取 gene id 2 symbol
    _index = {}
    
    index_file = '../ncbi/Homo_sapiens.gene_info'
    with open(index_file) as f:
        line = f.readline()
        for line in f:
            l = line.split('\t')
            _index[l[1]] = l[2]
    
    # 自由组合函数
    _combinations = lambda x, code='&': reduce(lambda x, y: [str(i)+code+str(j) for i in x for j in y], x)

    # Json file
    with open(rf) as f:
        r = 0
        for line in f: 
            r += 1
            if r%30000 == 0:
                print(r)
            
            _json = eval(line)
            pmid = _json['id']
            anns = _json['annotation'].split('&')
            
            for ann in anns:
                ann = ann.split('|')
                if ann != ['']:
                    if ann[3] not in annotation.keys():
                        annotation[ann[3]] = []
                        annotation[ann[3]].append((ann[2],ann[4]))
                    else:
                        annotation[ann[3]].append((ann[2],ann[4]))

# gene id
            if node_a in annotation.keys():
                for a in annotation[node_a]:
                    a = a[1].split(';')
                    for _a in a:
                        if _a != '':
                            _a = re.sub(u"\\(.*?\\)", "",_a)
                            if _index.get(_a) != None:
                                tem_a.append(_a)
                                if count_a.get(_a) == None:
                                    count_a[_a] = 1
                                else:
                                    count_a[_a] += 1
  
#chemical    
            if node_b in annotation.keys():
                for b in annotation[node_b]:
# chemical id
                    b = b[1].split(';')
                    for _b in b:
                        if _b != '':
                            tem_b.append(_b)
                            if count_b.get(_b) == None:
                                count_b[_b] = 1
                            else:
                                count_b[_b] += 1
                
#chemical name
#                    _b = str.lower(b[0])
#                    if _b != '':
#                        tem_b.append(_b)
#                        if count_b.get(_b) == None:
#                            count_b[_b] = 1
#                        else:
#                            count_b[_b] += 1
                     
                     
            single_pair.write('pmid: {0}\n'.format(pmid))

# 自由组合            
            tmp_dir = {}            
            if len(tem_a) != 0:
                pair_aa = list(combinations(tem_a, 2))
                for aa in pair_aa:
                    l = list(aa)
                    l.sort()
                    aa = tuple(l)
                    key = (aa, 'Gene-Gene')
                    if key not in tmp_dir.keys():
                        tmp_dir[key] = 1
                    else:
                        tmp_dir[key] += 1
                    if key[0][0] != key[0][1]:
                        if all_pair.get(key) == None:
                            all_pair[key] = 1
                        else:
                            all_pair[key] += 1                   
                for k in tmp_dir.keys():
                    single_pair.write('{0}\t{1}\t{2}\t{3}\n'.format(k[0][0], k[0][1], k[1], tmp_dir[k]))     
            else:
                pair_aa = []
            tmp_dir = {}
                        
            if len(tem_b) != 0:
                pair_bb = list(combinations(tem_b, 2))
                for bb in pair_bb:
                    l = list(bb)
                    l.sort()
                    bb = tuple(l)
                    key = (bb, 'Chemaical-Chemical')
                    if key not in tmp_dir.keys():
                        tmp_dir[key] = 1
                    else:
                        tmp_dir[key] += 1
                    if key[0][0] != key[0][1]:
                        if all_pair.get(key) == None:
                            all_pair[key] = 1
                        else:
                            all_pair[key] += 1
                for k in tmp_dir.keys():
                        single_pair.write('{0}\t{1}\t{2}\t{3}\n'.format(k[0][0], k[0][1], k[1], tmp_dir[k]))
            else:
                pair_bb = []
            tmp_dir = {}
            
            if len(tem_b) != 0 and len(tem_a) != 0:
                for ab in _combinations([tem_a, tem_b]):
                    key = (tuple((ab.split('&')[0], ab.split('&')[1])), 'Gene-Chemical')
                    if key not in tmp_dir.keys():
                        tmp_dir[key] = 1
                    else:
                        tmp_dir[key] += 1
                    if key[0][0] != key[0][1]:
                        if all_pair.get(key) == None:
                            all_pair[key] = 1
                        else:
                            all_pair[key] += 1
                for k in tmp_dir.keys():
                    single_pair.write('{0}\t{1}\t{2}\t{3}\n'.format(k[0][0], k[0][1], k[1], tmp_dir[k]))
            tmp_dir = {}
            single_pair.write('\n')
            
            tem_a = []
            tem_b = []
            annotation = {}
    print(r)
    
# Chemical_id to name
# 正在将 chemical 和 gene id 转为 name
# 读取 MeSH database
    mesh_index = {}
    tree = ET.parse(mesh)
    root = tree.getroot()
    for record in root.findall('DescriptorRecord'):
        k = record.find('DescriptorUI').text
        for string in record.find('DescriptorName'):
            v = string.text
        mesh_index[k] = v

# 读取 CHEBI database
    ebi_index = {}
    with open(ebi) as f:
        line = f.readline()
        for line in f:
            l = line.split('\t')
            ebi_id = l[2][6:]
            name = l[5]
            if name != 'null':
                ebi_index[ebi_id] = name  

# 读取 suplementary term
    supp_index = {}
    tree_supp = ET.parse(supp)
    root = tree_supp.getroot()
    for record in root.findall('SupplementalRecord'):
        k = record.find('SupplementalRecordUI').text
        for string in record.find('SupplementalRecordName'):
            v = string.text
        supp_index[k] = v
        
# 写文件            
    all_pair_file = '{0}/{1}_all_pair'.format(out, rf.split('/')[-1])
    file_node_a = '{0}/{1}_{2}_nodes'.format(out, rf.split('/')[-1], node_a)
    file_node_b = '{0}/{1}_{2}_nodes'.format(out, rf.split('/')[-1], node_b)
#    gene_list = '{0}/{1}_GeneList.txt'.format(out, rf.split('/')[-1])
    GeneSymbol = '{0}/{1}_GeneSymbol'.format(out, rf.split('/')[-1])
    ChemicalName = '{0}/{1}_ChemicalName'.format(out, rf.split('/')[-1])
#    mismatch = open('{0}/{1}_ChemicalName_mismatch'.format(out, rf.split('/')[-1]), 'w')

    with open(all_pair_file, 'w') as wf:
        wf.write('node_a\tnode_b\tinteracts\tedge_count\n')
        for k in sorted(all_pair, key=lambda x: all_pair[x], reverse=True):
            wf.write('{0}\t{1}\t{2}\t{3}\n'.format(k[0][0], k[0][1], k[1], all_pair[k]))
    with open(file_node_a, 'w') as wf:
        wf.write('id\tnode_type\tcount\n')
        for k in sorted(count_a, key=lambda x: count_a[x], reverse=True):
            wf.write('{0}\t{1}\t{2}\n'.format(k, node_a, count_a[k]))
    with open(file_node_b,'w') as wf:
        wf.write('id\tnode_type\tcount\n')
        for k in sorted(count_b, key=lambda x: count_b[x], reverse=True):            
            wf.write('{0}\t{1}\t{2}\n'.format(k,node_b, count_b[k]))
    with open(ChemicalName, 'w') as wf:
        wf.write('id\tchemical_name\n')
        for _id in count_b.keys():
            _name = ' '
            if ':' in _id:
                _id = _id.split(':')[1]
            if mesh_index.get(_id) != None:
                _name = mesh_index[_id]
            if ebi_index.get(_id) != None:
                _name = ebi_index[_id]
            if supp_index.get(_id) != None:
                _name = supp_index[_id]
#            if _name == ' ':
#                mismatch.write('{0}\n'.format(_id))
            wf.write('{0}\t{1}\n'.format(_id, _name))
            
#    mismatch.close()
    single_pair.close()
    with open(GeneSymbol, 'w') as f:
        f.write('id\tGeneSymbol')
        for _id in count_a.keys():
            if _index.get(_id) != None:
                f.write('{0}\t{1}\n'.format(_id, _index[_id]))

#    with open(gene_list, 'w') as wf:
#        for k in count_a.keys():
#            _k = re.sub(u"\\(.*?\\)", "",k)
#            wf.write('{0}\n'.format(_k))
    
# 分别截取前100, 50行用于做图
    os.system('head -101 {0} > {0}_100'.format(all_pair_file))
    os.system('head -51 {0} > {0}_50'.format(all_pair_file))    

# 汇总所有结点的 度 用于绘图    
    os.system('cat {0} {1} > {2}/nodes_count'.format(file_node_a, file_node_b, out))  
    print ('Done')
                        
if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('-r', '--rfile', help='data source, default: ../result/search_pubmed_index_Hepatitis_B',
                       dest='rf', required=False, default='../result/search_pubmed_index_Human_cytomegalovirus_infection')
    parse.add_argument('-o', '--outpath', help='save path, defaule ../result',
                        dest='out', required=False, default='../result')
                        
    parse.add_argument('-a', '-pair_a', help=' a nodes name, default: Gene, Optional: "SNP", "Disease", "Gene", "Chemical", "Species"',
                       dest='a', required=False, default='Gene')
    parse.add_argument('-b', '-pair_b', help=' b nodes name, default: Chemical, Optional: "SNP", "Disease", "Gene", "Chemical", "Species"',
                       dest='b', required=False, default='Chemical')     
                       
    parse.add_argument('-i', '--IndexFile', help='gene_id index file, default: ../ncbi/Homo_sapiens.gene_info',
                       dest='IndexFile', required=False, default='../ncbi/Homo_sapiens.gene_info')
                       
    parse.add_argument('-e', '--ebi', help='EBI database, default: ../ncbi/compounds.tsv',
                       dest='EBI', required=False, default='../ncbi/compounds.tsv')
    parse.add_argument('-m', '--mesh', help='MeSH database, default: ../ncbi/desc2019',
                       dest='MeSH', required=False, default='../ncbi/desc2019')
    parse.add_argument('-s', '--supp', help='MeSH database, default: ../ncbi/supp2015',
                       dest='supp', required=False, default='../ncbi/supp2015')
    args = parse.parse_args()
    
    if not os.path.exists(args.out):
        os.system('madir {0}'.format(args.out))
    
    Result_pair(args.rf, args.out, args.a, args.b, args.IndexFile, args.MeSH, args.EBI, args.supp)
