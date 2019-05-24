# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 11:46:38 2019

@author: yaoxinzhi
"""

'''
该代码用于将json 文件按 size分割为若干个文件
'''
import argparse
import os


def split(rf, path, size):
    count = 0
    num = 1
    wf = open('{0}/{1}'.format(path, num), 'w')
    with open(rf) as f:
        line = f.readline()
        while line!= '':
            wf.write(line)
            line = f.readline()
            count += 1
            if count % 100000 == 0:
                print (count)
            if count%size ==0:
                num += 1
                wf.close()
                wf = open('{0}/{1}'.format(path, num), 'w')
    wf.close()
    print ('Done')
    
if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('-f', '--file', help='Json file used to build the index,default: ../data/pubtator2json_split',
                   dest='file', default= '../data/pubtator2json_split')
    parse.add_argument('-p', '--path', help='json file path used to build the index, default: ../data/pubtator2json_yxz_split',
                       dest='path', default='../data/pubtator2json_yxz_split')
    parse.add_argument('-s', '--size', help='split size, default: 1000000',
                       dest='size', type=int, default= 1000000)
                       
                       
    args = parse.parse_args()
    if os.path.exists(args.path):
        os.system('rm {0} -fr'.format(args.path))
    if not os.path.exists(args.path):
        os.system('mkdir {0}'.format(args.path))
    split(args.file, args.path, args.size)