#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 11:05:20 2019

@author: yaoxinzhi
"""
import nltk
import argparse
import time
import os

'''
解析 json文件 text annotation
text 分句 并得到每句话的 offset信息
annotation 或取每个注释的 offset信息
匹配每个sentence中如果有两个及以上的注释 则save并且计算注释新的offset信息
'''
# 需要尝试 pubtator 的 offset 信息是否包括标题 妈的 包括 又要改


def co_sentence(json_file, outfile):
    ATTR_LIST = ['Gene', 'SNP', 'Chemical']
    STEP = 0
    sent_offset_dic = {}
    annotation_list = []
    rfile = open(outfile, 'w')
    time_start = time.time()
#    SENTENCE_COUNT=0
#    if os.path.exists('SENT_COUNT_ParsingTree'):
#        os.system('rm SENT_COUNT_ParsingTree')
#    SENT_COUNT = open('SENT_COUNT_ParsingTree', 'a')
    
    with open(json_file) as f:
        for line in f:
            
            STEP += 1
            if STEP % 30000 ==0:
                print ('STEP {0}'.format(STEP))
#            if STEP ==200:
#                break
            
            l = line.strip()
            # 解析 每一行 json 文件
            _json = eval(l)
            
            sent_offset_dic = {}
            result_list = []
            _text = _json['text']
            _id = _json['id']
            _title = _json['title']
            _annotation = _json['annotation']
            
            # title 和 text 分句 并计算sentence 的 offset 信息
            sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
            title_sent_list = sent_tokenizer.tokenize(_title)
            text_sent_list = sent_tokenizer.tokenize(_text)
            sent_list = title_sent_list + text_sent_list
            
            offset = 0
            for index, sent in enumerate(sent_list):
                _key = (offset, offset+len(sent))
                _sent = sent_list[index]
                offset = offset + len(sent) + 1
                sent_offset_dic[_key] = _sent

            # annotation 位置信息提取
            annotation_list = []
            ann_list = [ann for ann in _annotation.split('&')]
            
                # 判断text中我们关注的annotation条数小于2跳过进行下一篇text
            for a in ann_list:
                _ann = a.split('|')
                if _ann != ['']:
                    if _ann[3] in ATTR_LIST:
                        annotation_list.append(a)
            if len(annotation_list) == 0:
                continue         
            
            reverse_annotation = annotation_list[::-1]

            _key = sorted(sent_offset_dic.keys(), key=lambda x:x[0])
            
            # 遍历sentence
            # 这里复杂度待优化 匹配过的 annotation 重复循环 pop
            for k in _key:

                temp_dic = {}
                temp_dic['sentence'] = sent_offset_dic[k]
                temp_dic['annotation'] = []
                for index,a in enumerate(reverse_annotation):
                    ann = a.split('|')
                    if (int(ann[0]) == k[0]) :
                        if int(ann[1]) < k[1]: 
                            _offset = sent_offset_dic[k].find(ann[2])
                            temp_dic['annotation'].append('{0}|{1}|{2}|{3}|{4}'.format(_offset, _offset+len(ann[2]), ann[2], ann[3], ann[4]))                   
                            reverse_annotation.pop(index)
                        if int(ann[1]) == k[1]:
                            
                            _offset = sent_offset_dic[k].find(ann[2])
                            temp_dic['annotation'].append('{0}|{1}|{2}|{3}|{4}'.format(_offset, _offset+len(ann[2]), ann[2], ann[3], ann[4]))
                            reverse_annotation.pop(index)
                    else:
                        if int(ann[0]) > k[0]:
                            
                            if int(ann[1]) == k[1]:
    
                                _offset = sent_offset_dic[k].find(ann[2])
                                temp_dic['annotation'].append('{0}|{1}|{2}|{3}|{4}'.format(_offset, _offset+len(ann[2]), ann[2], ann[3], ann[4]))
                                reverse_annotation.pop(index)
                            else:
                                if int(ann[1]) < k[1]:
    
                                    _offset = sent_offset_dic[k].find(ann[2])
                                    temp_dic['annotation'].append('{0}|{1}|{2}|{3}|{4}'.format(_offset, _offset+len(ann[2]), ann[2], ann[3], ann[4]))
                                    reverse_annotation.pop(index)
                # 如果句子包含注释数大于2 存入result_list
                # 考虑后期可能需要注释数目只有一个的句子来coreference 所以保留
                if len(temp_dic['annotation']) > 2:
                    result_list.append(temp_dic)
#                    SENTENCE_COUNT += 1

            # save
            if result_list != []:
#                rfile.write('id\t{0}\n'.format(_id))
#                rfile.write('title\t{0}\n'.format(_title))
#                rfile.write('text\t{0}\n'.format(_text))
                for sent in result_list:
                    rfile.write('sentence\t{0}\n'.format(sent['sentence']))
                    for _ann in sent['annotation']:
                        rfile.write('annotation\t{0}\n'.format(_ann))
                    rfile.write('\n')
                rfile.write('\n')
#    SENT_COUNT.write('\n{0}\n'.format(SENTENCE_COUNT))

    rfile.close()
#    SENT_COUNT.close()
    time_end = time.time()
    print ('totally Abstract: {0} totally cost: {1}'.format(STEP, time_end-time_start))


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('-r', '--rfile path', help='data source path',
                       dest='rf', required=True)
    parse.add_argument('-w', '--wfile path', help='save file path ',
                        dest='wf', required=True)
    args = parse.parse_args()
    
    file_list = os.listdir(args.rf)
    if not os.path.exists(args.wf):
        os.system('mkdir {0}'.format(args.wf))
    
    for _file in file_list:
        rf = '{0}/{1}'.format(args.rf, _file)
        wf = '{0}/{1}_coSentence'.format(args.wf, _file)
        co_sentence(rf, wf)
        
    print ('Done')