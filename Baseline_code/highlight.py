# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 18:02:17 2019

@author: yaoxinzhi
"""

'''
该代码用于给单列的 id 文件 添加一列 H 
用于画图时 highlight 
'''

import argparse

def Highlight(rf):
    wf = '{0}_H'.format(rf)
    with open(wf, 'w') as wf:
        with open(rf) as f:
            wf.write('id\tHighlight\n')
            for line in f:
                l = line.strip()
                wf.write('{0}\tH\n'.format(l))
    print ('Done')
                        
if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('-r', '--rfile', help='data source',
                       dest='rfile', required=True)
            
    args = parse.parse_args()
    Highlight(args.rfile)