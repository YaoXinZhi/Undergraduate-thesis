# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 19:24:22 2019

@author: yaoxinzhi
"""

import argparse
import os

'''
该代码用于ElasticSearch搜索结果整理， 提取结果需要部分
目前版本用于提取 title 和 abstrace 
预计用于后期 stanford parser 处理
'''

def Result_processing(rf, att, outfile):
#    result = {}
#    att_list = []
    
    wf = open(outfile, 'w')
    with open(rf) as f:
        for line in f:
            l = line.strip().split()
            if len(l) == 0 or len(l) == 2 :
                pass
            else:
                _json = eval(line)
                _title = _json['title'].replace('[', '').replace(']', '')
                text = _json['text']
#                pmid = _json['id']
#                anns = _json['annotation'].split('&')

# get annotation                
#                for ann in anns:
#                    ann = ann.split('|')
#                    if ann != ['']:
#
#                        if ann[3] not in result.keys():
#                            result[ann[3]] = []
#                            result[ann[3]].append('{0}-{1}-{2}'.format(ann[0], ann[1], ann[4]))
#                        else:
#                            result[ann[3]].append('{0}-{1}-{3}'.format(ann[0], ann[1], ann[4]))
                            
#                    for k in result.keys():
#                        wf.write('{0}:\n'.format(k))
#                        for v in result[k]:
#                            wf.write('{0}\t'.format(v))
#                        wf.write('\n')
#                wf.write('\n')
#                    print (result.keys())

                            
#                if att in result.keys():
#                    wf.write('pmid: {0}\n'.format(pmid))
#                    wf.write('abstract:\t{0}\n'.format(text))
#                    wf.write('{0}:\t'.format(att))
#                    for value in result[att]:
#                        wf.write('{0}\t'.format(value))
#                        att_list.append(value.split('-')[2])
#                    wf.write('\n')
#                    wf.write('\n')
#                result = {}
#    att_list = set(att_list)
    
# 提取 title 和 abstract
            wf.write('title\t{0}\n'.format(_title))
            wf.write('abstract\t{0}\n'.format(text))
            wf.write('\n')
                            
    print ('Done')
                        
if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('-r', '--rfile', help='data source, default: ../result/search_pubmed_PancreaticCancer',
                       dest='rf', required=False, default='../result/search_pubmed_PancreaticCancer')
    parse.add_argument('-o', '--outpath', help='save path, defaule ../result',
                        dest='out', required=False, default='../result')
    parse.add_argument('-a', '-attributes', help=' attrivutes you need, default: Gene, Optional: "SNP", "Disease", "Gene", "Chemical", "Species"',
                       dest='att', required=False, default='Gene')
    args = parse.parse_args()
    if not os.path.exists(args.out):
        os.system('madir {0}'.format(args.out))
#    wf_all = '{0}/{1}_{2}'.format(args.out, args.rf.split('/')[-1], args.att)
#    wf_list = '{0}/{1}_{2}List'.format(args.out, args.rf.split('/')[-1], args.att)
    outfile = '{0}/{1}_Abstract'.format(args.out, args.rf.split('/')[-1])
    Result_processing(args.rf, args.att, outfile)