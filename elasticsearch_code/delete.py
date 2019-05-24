# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 12:59:21 2019

@author: yaoxinzhi
"""
'''
该代码用于 ElaticSearch 删除索引
'''
import argparse

from elasticsearch import Elasticsearch

def delete(_index):
    _es = Elasticsearch()
    _es.indices.delete(index=_index, ignore=[400, 404])
    print ('成功删除索引：{0}'.format(_index))

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('-i', '--index', help='index name to be deleted, default: pubmed_yao',
                       dest='index', required=False, default='pubmed_yao')
    args = parse.parse_args()
    delete(args.index)