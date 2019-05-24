# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 18:37:37 2019

@author: yaoxinzhi
"""

'''
该代码用于分割出一个size的文件 用于try try try
'''
import argparse

def split_file(rfile, wfile, size):
    count = 0
    with open(wfile, 'w') as wf:
        with open(rfile) as rf:
            for line in rf:
                count += 1
                wf.write(line)
                if count > size:
                    break
    print ('Done')
if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('-r', '--rfile', help='read file, default: "bioconcepts2pubtator_offsets"',
                       dest='rfile', default= 'bioconcepys2pubtator_offsets')
    parse.add_argument('-w', '--wfile', help='write file, default: pubtator_split',
                       dest='wfile', default= 'pubtator_split')
    parse.add_argument('-s', '--size', help='split size, default: 50000',
                       dest='size', type=int, default= 50000)
    args = parse.parse_args()
    split_file(args.rfile, args.wfile, args.size)
    
