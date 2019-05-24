#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 01:25:57 2019

@author: yaoxinzhi
"""

'''
该代码用于将 
elbo_result 转化为 ex_entity.py 可处理的
vae_result 格式
'''

import argparse

def main(rf, DepPath, out):
    wf = open(out, 'w')
    # 读取vae的结果
    vae_result = []
    with open(rf) as f:
        for line in f:
            l = line.strip()
            vae_result.append(l)
    vae_result = set(vae_result)
    # 抽提具有该路径的DepPath 完整信息
    count = 0
    with open(DepPath) as f:
        for line in f:
            count += 1
            if count % 5000 == 0:
                print (count)
            l = line.strip().split()
            if l[0] in vae_result:
                wf.write('{0}\n'.format(line.strip()))
    wf.close()
    

if __name__ == '__main__':
    
    parse = argparse.ArgumentParser()
    parse.add_argument('-r', '--rf', help='source file name',
                       dest='rf', required=True)
    parse.add_argument('-w', '--wf', help='output file name',
                       dest='wf', required=True)
    args = parse.parse_args()
    
    DepPath = '../data/chemical_gene_DepPath'
    
    main(args.rf, DepPath, args.wf)
    
    