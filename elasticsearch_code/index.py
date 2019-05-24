# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 23:02:11 2019

@author: yaoxinzhi
"""

from elasticsearch import Elasticsearch
from config import config as cf
from elasticsearch.helpers import bulk
import json
import os
import time
import argparse

es = Elasticsearch()
def create_index(client, _index):
    if not es.indices.exists(index= _index):
        client.indices.create(index=_index, body=cf.index_mapping)
    print ('successful to create mapping')
        
def index(datapath, index_name):
    
    create_index(es,index_name)
    with open(datapath,'r') as f:
        count=0
        error_count = 0
        actions=[]
        for line in f:
            _json = json.loads(line.strip())
            try:
                action = {"_index":index_name,
                                "_type":"PubMed",
                                "_id":_json["id"],
                                "_source":_json
                        }
            except KeyError:
                with open("error.log",'a') as wf:
                        wf.write(line)
                        error_count += 1
                continue

            actions.append(action)
            if len(actions)==3000:
                success, _ = bulk(es, actions, index = index_name)
                es.indices.refresh(index=index_name)
                count += success
                actions=[]
                print("成功插入 {0} 篇文本！".format(count))
        success, _ = bulk(es, actions, index = index_name)
        es.indices.refresh(index=index_name)
        count += success
        print("成功插入 {0} 篇文本！".format(count))
        if error_count != 0:
            print ('{0} errors have occurred'.format(error_count))
        
if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('-f', '--file', help='Json file used to build the index,default: ../data/pubtator2json_split',
                   dest='file', default= '../data/pubtator2json_split')
    parse.add_argument('-p', '--path', help='json file path used to build the index, default: ../data/pubtator2json_yxz_split',
                       dest='path', default='../data/pubtator2json_yxz_split')
    parse.add_argument('-i', '--index', help='the index name of you want to index, default: pubmed_yao',
                       dest='index_name', default='pubmed_yao')
    parse.add_argument('-o', '-option', help='Specify that you want to build an index with a single file or the path included myltiple files, default: p, options: "p" -> path, "s" -> single file',
                       dest='option', default='p')
                
    args = parse.parse_args()
    if args.option == 'p':
        file_list = os.listdir(args.path)
        while len(file_list) != 0:
            file = file_list.pop()
    #        print ('{0}/{1}'.format(args.path, file))
    #        file_name = '{0}/{1}'.format(args.path, file)
            try:
                print(file_list)
                print ('{0}/{1}'.format(args.path, file))
                index('{0}/{1}'.format(args.path, file), args.index_name)
                with open('log', 'a') as wf:
                    wf.write('{0}\n'.format(file))
                print("I'm sleeping")
                time.sleep(180)
                
            except:
                file_list.append(file)
                print("I'm sleeping")
                time.sleep(180)
    if args.option == 's':
        index(args.file, args.index_name)
    