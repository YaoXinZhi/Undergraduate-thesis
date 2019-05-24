#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 10:29:40 2019

@author: yaoxinzhi
"""

'''
该代码用于 通过id 匹配 pubtator 注释的 chemical 和 gene id 到 pharmGKB id
mesh --> Pubchem CID --> pharmGKB id

entrze id--> ensembl id --> pharmGKB id
'''

def  main(CID_MeSH, pharm_CID):
    
    # get mesh 2 cid
    with open(CID_MeSH) as f:
        cid_mesh = {}
        for line in f:
            l = line.strip().split('\t')
            cid_mesh[l[0]] = []
            for i in l[1:]:
                cid_mesh[l[0]].append(i.replace(' ', '-'))
    print (len(cid_mesh))
                
    # get cid 2 pharmGKB 
    with open(pharm_CID) as f:
        cid_pharm = {}
        for line in f:
            l = line.strip().split('\t')
#            print (l)
            if len(l[6].split(',')) > 1:
                for i in l[6].split(','):
                    if 'PubChem Compound' in i:
                        cid = i[18:-1]
                        cid_pharm[cid] = l[0]
            else:
                if 'PubChem Compound' in l[6]:
                    cid = l[3][17:]
                    cid_pharm[cid] = l[0]
        
    print (len(cid_pharm))
                

if __name__ == '__main__':
    
    CID_MeSH = '../database/pubchem/CID-MeSH'
    pharm_CID = '../database/pharmGKB/chemicals/chemicals.tsv'
    
    main(CID_MeSH, pharm_CID)