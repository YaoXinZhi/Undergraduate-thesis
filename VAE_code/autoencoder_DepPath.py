#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 09:27:02 2019

@author: yaoxinzhi
"""

import os
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

'''
目前 长度不足的依存路径在 储存依存路径时用 'padding' 填充
    单词和依存关系共同初始化为 embedding
'''
# 路径长度分布 {5: 431, 6: 65, 7: 96, 8: 38, 9: 1}
DEPPATH_SIZE = 7
EMBEDDING_SIZE = 100
BATCH_SIZE = 15
EPOCH = 15
LR = 0.05
LATENT_CODE_NUM = 32   
log_interval = 10
SAVE_PATH = 'autoencoder_mode'

def get_depPath_list(rf):
    # 从文件中读取所有依存路径
    # 读取 tab 分割的 第一列为start_ent|...|end_ent  长度用 padding 填充为0
    # 返回 包含  start_ent|...|end_ent 的列表
    DepPath_list = []
    with open(rf) as f:
        for line in f:
            l = line.split('\t')
            path = '|'.join([vac for vac in l[0].split('|')[1: -1]]).lower()
            # 删除重复通路
            if len(path.split('|')) < DEPPATH_SIZE:
                for i in range(0, DEPPATH_SIZE-len(path.split('|'))):
                    path += '|padding'
            if path not in DepPath_list:
                DepPath_list.append(path.lower())
    return DepPath_list

def get_embedding(DepPath_list):
    '''
    输入依存路径列表
    返回numpy二维数组 每条依存路径的编号
    DepPath_COUNT * DEPPATH_SIZE
    和
    每个词编号
    
    返回包含 embedding_tensor 的列表  
        word_to_id 需要从 0 开始编号
        embeds长度需要+1
    '''
    # 构建 vab2index 的字典
    word_to_id = {}
    count = 0
    for path in DepPath_list:
        # 删除 start_ent 和 end_ent
        for vab in path.split('|'):
            if not word_to_id.get(vab):
                word_to_id[vab] = count
                count += 1
    path_matrix = np.array([[word_to_id[vac] for vac in path.split('|')] for path in DepPath_list])
    
    return word_to_id, path_matrix

def index2embedding(word_to_id, index_matrix):
    '''
    该代码用于将 BATCH_SIZE * DEPPATH_SIZE 的 index 二维Tensor 
    转换为 BATCH_SIZE * DEPPATH_SIZE * EMBEDDING_SIZE 的 embedding 三维数组
    '''
    # 分配embedding
    embedding_dic = {}
    embeds = nn.Embedding(len(word_to_id.keys())+1, EMBEDDING_SIZE, padding_idx=word_to_id['padding'])
    
    id2vac = {word_to_id[vac]: vac for vac in word_to_id.keys()}
    
    
    for k in word_to_id.keys():
        lookup_tensor = torch.tensor([word_to_id[k]], dtype=torch.long)
        embed = embeds(lookup_tensor)
        embedding_dic[k] = embed
        
    embedding_matrix = [[list(embeds(torch.tensor([word_to_id[id2vac[int(index)]]], dtype=torch.long))[0].data.numpy()) for index in list(path)] for path in index_matrix[0].numpy()]
    
    return torch.from_numpy(np.array(embedding_matrix))

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
                nn.Linear(DEPPATH_SIZE * EMBEDDING_SIZE, 128),
                nn.Tanh(),
                nn.Linear(128, 64),
                nn.Tanh(),
                nn.Linear(64, 12),
                nn.Tanh(),
                nn.Linear(12, 3)
                )
        self.decoder = nn.Sequential(
                nn.Linear(3, 12),
                nn.Tanh(),
                nn.Linear(12, 64),
                nn.Tanh(),
                nn.Linear(64, 128),
                nn.Tanh(),
                nn.Linear(128, DEPPATH_SIZE * EMBEDDING_SIZE)
                )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

autoencoder = AutoEncoder()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()


def main():
    
    loss_dic = {}
    
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)
        
    
    flg_file = '../data/flg_path'
    
    DepPath_list = get_depPath_list(flg_file)
    word_to_id, path_matrix = get_embedding(DepPath_list)
    # 转化为 tensor
    path_matrix = torch.from_numpy(path_matrix)
    
    # minibatch
    torch_dataset = Data.TensorDataset(path_matrix)
    loader = Data.DataLoader(
            dataset = torch_dataset,
            batch_size = BATCH_SIZE,
            shuffle = True,
            num_workers = 2,
            drop_last = True
            )
    
    # train step
    for epoch in range(EPOCH):
        for step, batch_data in enumerate(loader):
            
            
            batch_embedding = index2embedding(word_to_id, batch_data)
            
#            print ('Epoch: {0}, |Step: {1}, |batch_data: {2}'.format(epoch+1, step+1, batch_embedding))
            batch_embedding_x = Variable(batch_embedding.view(-1, DEPPATH_SIZE * EMBEDDING_SIZE)) # (batch_size, DEPPATH_SIZE * EMBEDDING_SIZE)
            batch_embedding_y = batch_embedding.view(-1, DEPPATH_SIZE * EMBEDDING_SIZE) # (batch_size, DEPPATH_SIZE * EMBEDDING_SIZE)
            
#            print (batch_embedding_x.shape)
            
            
            encoded, decoded = autoencoder(batch_embedding_x)
            
            loss = loss_func(decoded, batch_embedding_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % 50 == 0:
                
                print ('Epoch: {0} | train loss: {1}'.format(epoch, loss.item()))
                torch.save(autoencoder, '{0}/autoencoder_model_{1}.pkl'.format(SAVE_PATH, epoch)) # entire net
                loss_dic['autoencoder_model_{0}.pkl'.format(epoch)] = float(loss.item())
    
    # 保留最佳模型
    key = sorted(loss_dic, key= lambda x: loss_dic[x])
    optimal_model = key[0]
    model_list = os.listdir(SAVE_PATH)
    for model in model_list:
#        print (model)
        if optimal_model != model:
            os.system('rm {0}/{1}'.format(SAVE_PATH, model))
    print ('\noptimal model: {0}/{1} | loss: {2}'.format(SAVE_PATH, optimal_model, loss_dic[optimal_model]))
    
if __name__ == '__main__':
    main()
