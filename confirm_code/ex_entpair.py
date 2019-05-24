#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 18:00:25 2019

@author: yaoxinzhi
"""

'''
该代码 用于 将实体对提出并标准化
chemical: name
gene: entrze id

因为TTD数据库 id 可以通过 entrze2uniprot 匹配 gene不筛选

'''

import re
import xml.etree.ElementTree as ET
import argparse

def ex_entpair(chemical_gene_DepPath, out_id):
    
#    index_file = '../database/ncbi/Homo_sapiens.gene_info'
    mesh = '../database/ncbi/desc2019'
    ebi = '../database/ncbi/compounds.tsv'
    supp = '../database/ncbi/supp2018'
    wf = open(out_id, 'w')
#    wf1= open(out_symbol, 'w')
    
    # get 全部结果的关系对
    id_dic = {}
#    symbol_dic = {}
#    count = 0
    with open(chemical_gene_DepPath) as f:
        for line in f:
            l = line.split('\t')
#            path = l[0]
            ent_1 = l[2].split('|')
            ent_2 = l[3].split('|')
            
            # 仅保存有ID的实体对
            if ent_1[4] != '' and ent_2[4] != '':
                # 对 pubtator注释的id进行一些标准化
                chem_id = ent_1[4]
                if ':' in chem_id:
                    chem_id = chem_id.split(':')[1]
                gene_id = re.sub(u"\\(.*?\\)", "",ent_2[4])
                if ';' in gene_id:
                    gene_id = gene_id.split(';')[0]
                
                id_pair = (chem_id, gene_id)
                
                if not id_dic.get(id_pair):
                    id_dic[id_pair] = 1
                else:
                    id_dic[id_pair] += 1
            
            # 通过symbol 计算实体对数目
#            chem_symbol = ent_1[2]
#            gene_symbol = ent_2[2]
#            symbol_pair = (chem_symbol, gene_symbol)
#            if not symbol_dic.get(symbol_pair):
#                symbol_dic[symbol_pair] = 1
#            else:
#                symbol_dic[symbol_pair] += 1
            
#     储存symbol_dic 结果
#    sym_key = sorted(symbol_dic, key = lambda x:symbol_dic[x], reverse=True)
#    for i in sym_key:
#        wf.write('{0}\t{1}\t{2}\n'.format(i[0], i[1], symbol_dic[i]))

                 
#     只储存id结果
#    id_key = sorted(id_dic, key = lambda x:id_dic[x], reverse=True)
#    for i in id_key:
#        wf.write('{0}\t{1}\t{2}\n'.format(i[0], i[1], id_dic[i]))
#    
    
#    print ('ex gene index')
#    # 读取数据库数据 用于 将gene id转换为 symbol
#    # ../database/ncbi/Homo_sapiens.gene_info
#    gene_index = {}
#    with open(index_file) as f:
#        line = f.readline()
#        for line in f:
#            l = line.split('\t')
#            gene_index[l[1]] = l[2].replace(' ', '-')
            
    print ('ex mesh')
    # ../database/ncbi/desc2019
    mesh_index = {}
    tree = ET.parse(mesh)
    root = tree.getroot()
    for record in root.findall('DescriptorRecord'):
        k = record.find('DescriptorUI').text
        for string in record.find('DescriptorName'):
            v = string.text
        mesh_index[k] = v.replace(' ', '-')
        
    print ('ex ebi ')
#    # ../database/ncbi/compounds.tsv
    ebi_index = {}
    with open(ebi) as f:
        line = f.readline()
        for line in f:
            l = line.split('\t')
            ebi_id = l[2][6:]
            name = l[5]
            if name != 'null':
                ebi_index[ebi_id] = name.replace(' ', '-')
                
    print ('supp2018')
    # ../database/ncbi/supp2018
    supp_index = {}
    tree_supp = ET.parse(supp)
    root = tree_supp.getroot()
    for record in root.findall('SupplementalRecord'):
        k = record.find('SupplementalRecordUI').text
        for string in record.find('SupplementalRecordName'):
            v = string.text
        supp_index[k] = v.replace(' ', '-')
        
    print('saving')
#    # save
#    wf.write('ent_pair\tchemical_symbol\tgene_symbol\tcount\n')
    # 按照 计数排序
    id_key = sorted(id_dic, key = lambda x:id_dic[x], reverse=True)    
    for  i in id_key:
        chem_id = i[0]
        gene_id = i[1]
        
        chem_symbol = 'None'
#        gene_symbol = 'None'
#        
        if mesh_index.get(chem_id):
            chem_symbol = mesh_index[chem_id]
        if ebi_index.get(chem_id):
            chem_symbol = ebi_index[chem_id]
        if supp_index.get(chem_id):
            chem_symbol = supp_index[chem_id]
#            
#        # gene 匹配不上的说明不是人类
#        if gene_index.get(gene_id):
#            gene_symbol = gene_index[gene_id]
            
        # 匹配不上 chemical name 的不要 
        if chem_symbol != 'None':
            wf.write('{0}\t{1}\t{2}\t{3}\n'.format((chem_id, gene_id), chem_symbol, gene_id, id_dic[i]))
#        
    wf.close()

if __name__ == '__main__':

    parse = argparse.ArgumentParser()
    parse.add_argument('-r', '--rfile path', help='Source file name',
                   dest='rf', required=True)
    parse.add_argument('-w', '--wf', help='output file name',
                        dest='w', required=True)
#    parse.add_argument('-w2', '--wf_symbol', help='wf_symbol file name',
#                   dest='wf2', required=True)
    args = parse.parse_args()
    
#    chemical_gene_DepPath = '../data/chemical_gene_DepPath'
#    wf = open('../result/ent_pair_id', 'w')
#    wf1= open('../result/ent_pair_symbol', 'w')
    ex_entpair(args.rf, args.w)
                   