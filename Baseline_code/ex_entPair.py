#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 11:16:16 2019

@author: yaoxinzhi
"""

'''
该代码用于提取 baseline1 两条线的 结果的实体对
输入文件格式为 _all_pair
    ent1_id    ent2_id    relationship    count
输出文件格式为 
    (chemical_id, gene_id)    chemical_name    gene_id    count
'''

import re
import xml.etree.ElementTree as ET
import argparse

def ex_entpair(rf, out):
    wf = open(out, 'w')
    
    mesh = '../ncbi/desc2019'
    ebi = '../ncbi/compounds.tsv'
    supp = '../ncbi/supp2018'
    
    
    with open(rf) as f:
        chemical_gene_count = {}
        for line in f:
            l = line.strip().split('\t')
            try:
                if l[2] == 'Gene-Chemical':
                    chem_id = l[1]
                    if ':' in chem_id:
                        chem_id = chem_id.split(':')[1]
                    gene_id = re.sub(u"\\(.*?\\)", "", l[0])
                    if ';' in gene_id:
                        gene_id = gene_id.split(';')[0]
                    pair = (chem_id, gene_id)
                    if not chemical_gene_count.get(pair):    
                        chemical_gene_count[pair] = int(l[-1])
                    else:
                        chemical_gene_count[pair] += int(l[-1])
            except:
                continue

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
        
    print ('saving')
    _key = sorted(chemical_gene_count, key = lambda x:int(chemical_gene_count[x]), reverse=True)
    for i in _key:
        chem_id = i[0]
        gene_id = i[1]
        
        chem_symbol = 'None'
        
        if mesh_index.get(chem_id):
            chem_symbol = mesh_index[chem_id]
        if ebi_index.get(chem_id):
            chem_symbol = ebi_index[chem_id]
        if supp_index.get(chem_id):
            chem_symbol = supp_index[chem_id]
            
#        if chem_symbol != 'None':
        wf.write('{0}\t{1}\t{2}\t{3}\n'.format((chem_id, gene_id), chem_symbol, gene_id, chemical_gene_count[i]))
#        
    wf.close()

if __name__ == '__main__':
    
    parse = argparse.ArgumentParser()
    parse.add_argument('-r', '--rfile path', help='Source file name',
                   dest='rf', required=True)
    parse.add_argument('-w', '--wf', help='output file name',
                        dest='w', required=True)
    args = parse.parse_args()
    
    ex_entpair(args.rf, args.w)