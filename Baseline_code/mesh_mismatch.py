# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 21:30:42 2019

@author: yaoxinzhi
"""

'''
该代码用于测试这些匹配不到的 seplementary term 他妈的到底属于那年的文件
 不匹配的存于 ../data/unindex
 seplementary 文件存于 ../ncbi/supp201*
 
 supp2013 3675/3715
 supp2014 3684/3715
 supp2015 3693/3715   
 supp2016 3642/3715   
 supp2017 3630/3715   
 supp2018 3623/3715
 supp2019 3508/3715
 
 肯定妈的哪年删了一些又加了一些 但我并不想管了！ 
 就用2015年的吧
'''



import argparse
import xml.etree.ElementTree as ET



def Chemical2Symbol(rfile, ofile, mesh):

    un_index = open('../result/unindex', 'w')

# 读取 MeSH database
    mesh_index = {}
    tree = ET.parse(mesh)
    root = tree.getroot()
    for record in root.findall('SupplementalRecord'):
        k = record.find('SupplementalRecordUI').text
        for string in record.find('SupplementalRecordName'):
            v = string.text
        mesh_index[k] = v
             
# 读取 chemical 列表
    chemical_id = []
    with open(rfile) as f:
        for line in f:      
            l = line.strip().split('\t')
            _id = l[0]
            chemical_id.append(_id)
    print (len(chemical_id))
    
# id to name
    count = 0
    with open(ofile, 'w') as wf:
        wf.write('id\tchemical_name\n')
        for _id in chemical_id:
            _name = ' '
            if mesh_index.get(_id) != None:
                _name = mesh_index[_id]
                count += 1
            if _name == ' ':
                un_index.write('{0}\n'.format(_id))
            wf.write('{0}\t{1}\n'.format(_id, _name))
    print (count)
    un_index.close()

    print ('Done')
        


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('-m', '--mesh', help='MeSH database, default: ../ncbi/supp2018',
                       dest='MeSH', required=False, default='../ncbi/supp2018')
    parse.add_argument('-r', '--rFile', help='Chemical id file',
                        dest='rfile', required=True)                      
    parse.add_argument('-o', '--OFile', help='save file',
                        dest='ofile', required=True)
    args = parse.parse_args()
    
    Chemical2Symbol(args.rfile, args.ofile, args.MeSH)