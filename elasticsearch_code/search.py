# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 12:58:51 2019

@author: yaoxinzhi
"""

'''
本代码用于ElasticSearch的检索
目前只支持 单个关键字检索
'''
'''
note:
1. filter 过滤 不评分 / query 评分 适合全文搜索
2. constant_score 一个不变的常量评分应用于所有匹配的文档 用来取代只要执行一个filter的bool查询
3. term 精确值匹配 terms / range 范围 / match 标准查询 multi_match / exists / missing
4. must/must_not/should/filter
5. explain 解释
6. _source 提取所需字段
'''

import argparse
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
import os
import time

#es = Elasticsearch(timeout=30, max_retries=10, retry_on_timeout=True)
es = Elasticsearch(timeout=30)


def Elastic_search(keyword, _index, fields, output):

# 设置 max_result_window
#    es.indices.put_settings(index=_index, body={'index':{'max_result_window':10000}})


    print ('\nindex:{2}, fields:{0}, query: "{1}"'.format(fields, keyword, _index))
    
# 搜索体   
# soft search
#    _query = {
#                'query': {
#                        'constant_score': {
#                        'filter':{
#                            "multi_match": {
#                                "query":                keyword,
#                                "type":                 "most_fields", 
#                                "fields":               fields,
#                                "tie_breaker":          0.3,
#                                "minimum_should_match": "30%"
#                                "operator": "and"
#                            }
#                        }
#                        }
#                }
#        }

# term search  
#    _query = {
#                'query': {
#                        'constant_score': {
#                        'filter':{
#                                'bool':{
#                                        'should':[
#                                                {"match_phrase":{"text":"pancreatic cancer"}},
#                                                #{"term":{"annotation": "snp"}}
#                                                ],
#                                        }
#                                }
#                        }
#                }
#        }
    

    _match = [{'match_phrase':{i: keyword}} for i in fields]
    _query = {'query': {'constant_score':{'filter':{'bool':{'should': _match}}}}} 
#    print (_query)
   
# search 语句  
# 分页匹配
    count = 0
    with open(output, 'w') as wf:
        for i in scan(es, query=_query, size=10000, index=_index, doc_type='PubMed'):
            count += 1
            wf.write(str(i['_source']))
            wf.write('\n')
#            if count % 60000 == 0 :
#                print ('{0} matching results found'.format(count))
#                print ("I sleep for 60s")
#                time.sleep(60)
     
# 不分页匹配
#    _searched = es.search(index=_index, doc_type='PubMed', body=_query, size =10000)
#    
#    count = 0
#    with open(output, 'w') as wf:
#        for hit in _searched['hits']['hits']:   
#            count += 1
#            wf.write(str(hit['_source']))
#            wf.write('\n')
#            wf.write('\n')
#            print ('Score: {0}'.format(hit['_score']))
#            print (hit['_source'], flush=True)         
 
   
    print ('\nSearch results--{1} are stored in file "{0}"'.format(output, count))

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('-s', '--sword', help='single keyword for search, -s "pancreatic cancer", default: "pancreatic cancer"',
                       dest='skeyword', required=False, default='pancreatic cancer')
    parse.add_argument('-f', '--fields', help='fields that need to be searched -f "title" "text" "annotation", default: ["title", "text", "annotation"] ',
                       dest='fields', nargs='+', type=str, default= ['title', 'text', 'annotation'])
    parse.add_argument('-i', '--index', help='Index for search, default: pubmed_yao',
                       dest='index', default= 'pubmed_yao')
    parse.add_argument('-o', '--output', help='output path, default: ../result',
                      dest='output', required=False, default='../result')        
    args = parse.parse_args()
    if not os.path.exists(args.output):
        os.system('mkdir {0}'.format(args.output))
    
    print ('_'.join(args.skeyword.split()))
    wf = '{0}/search_{1}_{2}'.format(args.output, args.index, '_'.join(args.skeyword.split()))    
    Elastic_search(args.skeyword, args.index, args.fields, wf)
#    print ('{0} documents in index'.format(es.count(index='pubmed_yao')['count']))