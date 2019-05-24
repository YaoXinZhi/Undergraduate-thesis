# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 13:01:48 2019

@author: yaoxinzhi
"""

'''
该代码用于gene列表文件和 name 转换
通过 Homo_sapiens.gene_info 文件
'''


import argparse
import re

def GeneIdConversion(index_file, GeneList_file, wf):
    index = {}
    gene_id = []

    
    with open(index_file) as f:
        line = f.readline()
        for line in f:
            l = line.split('\t')
            index[l[1]] = l[2]

    with open(GeneList_file) as f:
        for line in f:
            l = line.strip().split('\t')
            _id = re.sub(u"\\(.*?\\)", "",l[0])
            gene_id.append(_id)
            
    count = 0 
    with open(wf, 'w') as f:
        for _id in gene_id:
            if index.get(_id):
                f.write('{0}\t{1}\n'.format(_id, index[_id]))
            else:
#                print (_id)
                count += 1
    print (count)
    
    print ('Done')
            

                        
if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('-i', '--IndexFile', help='data source, default: ../ncbi/Homo_sapiens.gene_info',
                       dest='IndexFile', required=False, default='../ncbi/Homo_sapiens.gene_info')
    parse.add_argument('-g', '--GeneList', help='gene list file',
                       dest='Genelist', required=True)
    parse.add_argument('-o', '--OutFile', help='save file, defaule ../result',
                        dest='OutFile', required=False, default='../result')                
    args = parse.parse_args()
    
    GeneIdConversion(args.IndexFile, args.Genelist, args.OutFile)