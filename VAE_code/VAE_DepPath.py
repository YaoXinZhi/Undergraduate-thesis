#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 10:48:33 2019

@author: yaoxinzhi
"""

import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

'''
目前 长度不足的依存路径在 储存依存路径时用 'padding' 填充
    单词和依存关系共同初始化为 embedding
'''
# 路径长度分布 {5: 431, 6: 65, 7: 96, 8: 38, 9: 1}
DEPPATH_DIM = 7
EMBEDDING_DIM = 100
BATCH_SIZE = 15
EPOCH = 10
LR = 0.001
LATENT_DIM = 16  
H_DIM = 40
LATENT_DIM = 16
SHUFFLE = True
NUM_WORKERS = 2

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
            if len(path.split('|')) < DEPPATH_DIM:
                for i in range(0, DEPPATH_DIM-len(path.split('|'))):
                    path += '|padding'
            if path not in DepPath_list:
                DepPath_list.append(path.lower())
    return DepPath_list

def get_embedding(DepPath_list):
    '''
    输入依存路径列表
    返回numpy二维数组 每条依存路径的编号
    DepPath_COUNT * DEPPATH_DIM
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
    该代码用于将 BATCH_SIZE * DEPPATH_DIM 的 index 二维Tensor 
    转换为 BATCH_SIZE * DEPPATH_DIM * EMBEDDING_DIM 的 embedding 三维数组
    '''
    # 分配embedding
    embedding_dic = {}
    embeds = nn.Embedding(len(word_to_id.keys())+1, EMBEDDING_DIM, padding_idx=word_to_id['padding'])
    
    id2vac = {word_to_id[vac]: vac for vac in word_to_id.keys()}
    
    
    for k in word_to_id.keys():
        lookup_tensor = torch.tensor([word_to_id[k]], dtype=torch.long)
        embed = embeds(lookup_tensor)
        embedding_dic[k] = embed
        
    embedding_matrix = [[list(embeds(torch.tensor([word_to_id[id2vac[int(index)]]], dtype=torch.long))[0].data.numpy()) for index in list(path)] for path in index_matrix[0].numpy()]
    
    return torch.from_numpy(np.array(embedding_matrix))

class VAE(nn.Module):
      def __init__(self):
            super(VAE, self).__init__()
            self.encoder = nn.Sequential(
                    nn.Linear(DEPPATH_DIM * EMBEDDING_DIM, H_DIM),
                    nn.LeakyReLU(0.2),
                    nn.Linear(H_DIM, LATENT_DIM * 2)
                    )
            
            self.decoder = nn.Sequential(                
                  nn.Linear(LATENT_DIM, H_DIM),
                  nn.ReLU(0.2),
                  nn.Linear(H_DIM, DEPPATH_DIM * EMBEDDING_DIM),
                  nn.Sigmoid()
                  )

      def reparameterize(self, mean, log_var):
          samples = Variable(torch.randn(mean.size(0), mean.size(1)))
          z = mean + samples * torch.exp(log_var / 2)
          return z
      
      def forward(self, x):
          h = self.encoder(x)
          mean, log_var = torch.chunk(h, 2, dim=1)
          z = self.reparameterize(mean, log_var)
          out = self.decoder(z)
          return out, mean, log_var
      
      def sample(self, z):
          return self.decoder(z)
          
def loss_func(out, batch_data, mean, log_var):
    reconst_loss = F.binary_cross_entropy(out, batch_data, size_average=False)
    kl_divergence = torch.sum(0.5 * (mean **2 + torch.exp(log_var) - log_var - 1 ))
    total_loss = reconst_loss + kl_divergence
    return reconst_loss, kl_divergence, total_loss

def main():
#    dep_path = 'start_entity|dobj|inhibitor|nsubj|effects|nmod|end_entity'
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
            shuffle = SHUFFLE,
            num_workers = NUM_WORKERS,
            drop_last = False
            )
    
    vae = VAE()
    optimizer = optim.Adam(vae.parameters(), lr=LR)
    
    loss_list = []
    # train step
    count = 0 
    total_loss = 0
    for epoch in range(EPOCH):

        for step, batch_data in enumerate(loader):
#            count +=1 
            
            batch_embedding = index2embedding(word_to_id, batch_data) # ([15, 7, 100])
            
             # (batch_size, DEPPATH_SIZE * EMBEDDING_SIZE)
            batch_embedding_x = Variable(batch_embedding.view(-1, DEPPATH_DIM * EMBEDDING_DIM))
            batch_embedding_y = Variable(batch_embedding.view(-1, DEPPATH_DIM * EMBEDDING_DIM))
            
            # out [BATCH_SIZE, DEPPATH_SIZE * EMBEDDING_SIZE], mean [BATCH_SiZE, LATENT_DIM], log_var [BATCH_SIZE, LATENT_DIM]
            out, mean, log_var = vae.forward(batch_embedding_x)
            
            print ('out:\n{0}\nembedding:\n{1}\n'.format(out, batch_embedding))
            reconst_loss, kl_divergence, total_loss = loss_func(out, batch_embedding_y, mean, log_var)
            
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            if step % 10 == 0:
                count += 1
                loss_list.append(total_loss/len(torch_dataset))
                print ('step: {3} | reconst_loss: {0} | kl_divergence: {1} | total_loss: {2}'.format(reconst_loss, kl_divergence, total_loss, step))
                
    # 绘制 loss 曲线
    x1 = range(0, count)
    y1 = loss_list
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, '.-')
    plt.xlabel('Test loss vs. epoches')
    plt.ylabel('Test loss')
    plt.show()
    
if __name__ == '__main__':
    main()
