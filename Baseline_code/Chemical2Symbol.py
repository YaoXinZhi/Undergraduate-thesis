# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 14:58:10 2019

@author: yaoxinzhi
"""

'''
该代码用于将 pubtator 标注的 chemical id 转化为 chemical name
chemical 包括 MeSH 和 CHEBI 两类
CHEBI 数据为 EBI id  -- ../ncbi/compounds.tsv
MeSH 为 NCBI MeSH id -- ../ncbi/desc2019
MeSH_supp 为 ../ncbi/supp2015

是的 三个字典是可以写成一个的 我就是想看看是不是三个文件都有用
'''

import argparse
import xml.etree.ElementTree as ET



def Chemical2Symbol(rfile, ofile, mesh, ebi, supp):

    un_index = open('../result/unindex', 'w')

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
 
# 读取 chemical 列表
    chemical_id = []
    with open(rfile) as f:
        line = f.readline()
        for line in f:
            l = line.split('\t')
            _id = l[0]
            if ':' in _id:
                _id = _id.split(':')[1]
            chemical_id.append(_id)

# id to name
    count1 = 0
    count2 = 0
    count3 = 0
    with open(ofile, 'w') as wf:
        wf.write('id\tchemical_name\n')
        for _id in chemical_id:
            if mesh_index.get(_id):
                _name = mesh_index[_id]
                count1 += 1
            if ebi_index.get(_id):
                _name = ebi_index[_id]
                count2 += 1
            if supp_index.get(_id):
                _name = supp_index[_id]
                count3 += 1
            if _name == ' ':
                un_index.write('{0}\n'.format(_id))
            wf.write('{0}\t{1}\n'.format(_id, _name))
            _name = ' '
    print (count1)
    print (count2)
    print (count3)
    print ('{0} / {1}'.format(count1+count2+count3, len(chemical_id)))
    un_index.close()

    print ('Done')
        
if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('-e', '--ebi', help='EBI database, default: ../ncbi/compounds.tsv',
                       dest='EBI', required=False, default='../ncbi/compounds.tsv')
    parse.add_argument('-m', '--mesh', help='MeSH database, default: ../ncbi/desc2019',
                       dest='MeSH', required=False, default='../ncbi/desc2019')
    parse.add_argument('-s', '--supp', help='MeSH database, default: ../ncbi/supp2015',
                       dest='supp', required=False, default='../ncbi/supp2015')
    parse.add_argument('-r', '--rFile', help='Chemical id file',
                        dest='rfile', required=True)                      
    parse.add_argument('-o', '--OFile', help='save file',
                        dest='ofile', required=True)
    args = parse.parse_args()
    
    Chemical2Symbol(args.rfile, args.ofile, args.MeSH, args.EBI, args.supp)