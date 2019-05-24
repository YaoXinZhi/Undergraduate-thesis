# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 13:19:25 2018

@author: yaoxinzhi
"""

'''
test：
1. 分词器效果

'''

'''
notes:
1. type: long double date 不会被分析
2. 已经存在的 mapping 无法更改， 但可以直接增加新的域
'''

'''
本代码用于 ElasticSearch 创建索引
'''



class config():
    
    
    index_mapping = {
      "mappings": {
        "PubMed": {
          "properties": {
            "id":{
                  "type": "integer",
                  #"type": "long",
              },
            "title": {
              "type": "text",
              "analyzer": "english"
              #"analyzer": "seandard"
            },
            "text": {
              "type": "text",
              "analyzer": "english"
            },
             "annotation": {
              "type": "text",
              "analyzer": "english"
            }
          }
        }
      }
    }
