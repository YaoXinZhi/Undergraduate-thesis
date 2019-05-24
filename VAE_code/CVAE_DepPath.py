#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 16:28:30 2019

@author: yaoxinzhi
"""

'''
该代码为 CVAE 的改进版本
将 Embedding 写入网络参数 动态更新
'''
import os
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


#torch.manual_seed(1)
# 路径长度分布 {5: 431, 6: 65, 7: 96, 8: 38, 9: 1}

DEPPATH_DIM = 7
EMBEDDING_DIM = 100
CATEGORIES = 7
BATCH_SIZE = 15
EPOCH = 20
LR = 0.001
LATENT_DIM = 16  
SHUFFLE = True
NUM_WORKERS = 2
TRAIN_DATA = '../data/flg_path_11c'
SAVE_PATH = 'model'
TEST_DATA = '../data/chemical_gene_DepPath'
OUT = '../result/vae_pattern'


def get_depPath_list(rf):
    # 从文件中读取所有依存路径
    # 读取 tab 分割的 第一列为start_ent|...|end_ent  长度用 padding 填充为0
    # 返回 包含  start_ent|...|end_ent 的列表
    DepPath_list = []
    with open(rf) as f:
        for line in f:
            l = line.strip().split('\t')
            path = '|'.join([vac for vac in l[0].split('|')[1: -1]]).lower()
            label = l[-1]
            # 删除重复通路
            if len(path.split('|')) < DEPPATH_DIM:
                for i in range(0, DEPPATH_DIM-len(path.split('|'))):
                    path += '|padding'
            if path not in DepPath_list:
                DepPath_list.append((path.lower(), label))
    return DepPath_list

def get_depPath_list1(rf):
    # 从文件中读取所有依存路径
    # 读取 tab 分割的 第一列为start_ent|...|end_ent  长度用 padding 填充为0
    # 返回 包含  start_ent|...|end_ent 的列表
    DepPath_list = []
#    DepPath2line = {}
    count = 0
    with open(rf) as f:
        for line in f:
            count += 1
            l = line.strip().split('\t')
            if count % 50000 == 0:
                print('reading depPath: {0}'.format(count))
            path = '|'.join([vac for vac in l[0].split('|')[1: -1]]).lower()
#            label = l[-1]
            # 删除重复通路
#            print (path)
#            print (len(path.split('|')), DEPPATH_DIM)
            path_len = len(path.split('|'))
            if path_len < DEPPATH_DIM:
                for i in range(0, DEPPATH_DIM-len(path.split('|'))):
                    path += '|padding'
            if path_len < DEPPATH_DIM or path_len == DEPPATH_DIM:
                DepPath_list.append(path.lower())
    DepPath_list = list(set(DepPath_list))
#                DepPath2line[path.lower()] = line.strip()
    return DepPath_list

def get_embedding(path_data, test_data):
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
    DepPath_list = [i[0] for i in path_data]
    label_list = [i[1] for i in path_data]
    
    all_path_list = DepPath_list + test_data
    for path in all_path_list:
        # 删除 start_ent 和 end_ent
        for vab in path.split('|'):
            if not word_to_id.get(vab):
                word_to_id[vab] = count
                count += 1
                
    temp_list = []
    path_matrix = []
    for index, path in enumerate(DepPath_list):
        
        temp_list = [word_to_id[vac] for vac in path.split('|')]
#        print (temp_list)
        temp_list.append(int(label_list[index]))
        path_matrix.append(temp_list)
    path_matrix = np.array(path_matrix)
    
    temp_list = []
    test_matrix = []
    for index, path in enumerate(test_data):
        temp_list = [word_to_id[vac] for vac in path.split('|')]
        test_matrix.append(temp_list)
    test_matrix = np.array(test_matrix)
#    path_matrix = np.array([[word_to_id[vac] for vac in path.split('|') ] for index, path in enumerate(DepPath_list)])
    return word_to_id, path_matrix, test_matrix

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

class CVAE(nn.Module):
    def __init__(self, vocab_size, embedding_dim, word_to_id):
        
        super(CVAE, self).__init__()
        # 词向量
        self.embeddings = nn.Embedding(vocab_size+1, embedding_dim, padding_idx=word_to_id['padding'])
        
        # 二维卷积
        self.inference_net = nn.Sequential(
                nn.Conv2d(DEPPATH_DIM, 2 * LATENT_DIM, (EMBEDDING_DIM, 1), 2),
                nn.Conv2d(2 * LATENT_DIM, 4 * LATENT_DIM, 2, 2),
                nn.Conv2d(4 * LATENT_DIM, 8 * LATENT_DIM, 2, 2)
                )
        
        
        self.fc1 = nn.Linear(8 * LATENT_DIM, 2 * LATENT_DIM)
        self.fc12 = nn.Linear(LATENT_DIM, 2 * LATENT_DIM)
        
        self.fc2 = nn.Linear(2 * LATENT_DIM, 4 * LATENT_DIM)
        self.fc3 = nn.Linear(4 * LATENT_DIM, CATEGORIES)
        
        # 二维
        self.generative_net = nn.Sequential(
#                nn.ConvTranspose2d(8 * LATENT_DIM, 4 * LATENT_DIM, 1, 2),
                nn.ConvTranspose2d(LATENT_DIM, 4 * LATENT_DIM, 1, 2),
                nn.ConvTranspose2d(4 * LATENT_DIM, 2 * LATENT_DIM, 1, 2),
                nn.ConvTranspose2d(2 * LATENT_DIM, DEPPATH_DIM, (EMBEDDING_DIM, 1), 2),
                nn.Sigmoid()
                )
        
    def encoder(self, x):
        x = self.embeddings(x).unsqueeze_(-1)
        out1 = self.inference_net(x)
        out2 = out1.view(out1.size(0), -1)
        out3 = self.fc1(out2)
        mean, log_var = torch.chunk(out3, 2, dim=1)
        return mean, log_var
    
    def decoder(self, z):
        #分别对采样y 分类 和 重构
        # 分类
        z1 = F.tanh(self.fc12(z))
        z_class = F.relu(self.fc2(z1))
        z_class = self.fc3(z_class)
#        z_class = F.relu(self.fc2(z))
        # 重构
        z_ = z.view(z.size(0), z.size(1), 1, 1)
        logits = self.generative_net(z_)
        
        return logits, z_class
    
    def reparameterize(self, mean, log_var):
        # mean [batch_size, latent_size]
        # logvar [batch_size, latent_size]
        eps = torch.randn(mean.size(0), LATENT_DIM)
        # z [batch_size, latent_size]
        z = eps * torch.exp(log_var*.5) + mean
        return z
    def compute_loss(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        # 在这里加分类全链接
        out, out_class = self.decoder(z)
        # batch_size * 1 
#        class_vec = torch.max(F.softmax(out_class), 1)[1]
        return out, mean, log_var, out_class
        
    
def loss_func(out, batch_data, mean, log_var):
    # elbo = 重建损失 + KL散度
#    if out.size(0) == 1:
#        out = out.unsqueeze(0)
#    print (out.shape, batch_data.shape)
    reconst_loss = F.binary_cross_entropy_with_logits(out, batch_data)
    kl_divergence = torch.sum(0.5 * (mean **2 + torch.exp(log_var) - log_var - 1 ))
    total_loss = reconst_loss + kl_divergence
    # 分类预测损失
    
    return reconst_loss, kl_divergence, total_loss

def main():
#    dep_path = 'start_entity|dobj|inhibitor|nsubj|effects|nmod|end_entity'
    
    DepPath_list = get_depPath_list(TRAIN_DATA)
    
    DepPath_list1 = get_depPath_list1(TEST_DATA)
    
    
    
    word_to_id, path_matrix, test_matrix = get_embedding(DepPath_list, DepPath_list1)
    # 转化为 tensor
    path_matrix = torch.from_numpy(path_matrix)
    
    # minibatch
    torch_dataset = Data.TensorDataset(path_matrix)
    loader = Data.DataLoader(
            dataset = torch_dataset,
            batch_size = BATCH_SIZE,
            shuffle = SHUFFLE,
            num_workers = NUM_WORKERS,
            drop_last = True
            )
    
    cvae = CVAE(len(word_to_id.keys()), EMBEDDING_DIM, word_to_id)
    optimizer = optim.Adam(cvae.parameters(), lr=LR)
    # 预测损失用的损失函数
    loss_func_pred = nn.CrossEntropyLoss()
    
    
    loss_list = []
    # train step
    count = 0 
    loss_list_1 = []
    # train step
    for epoch in range(EPOCH):
        total_loss = 0
        for step, batch_data in enumerate(loader):
            # 拿数据
            data = batch_data[0].data.numpy()
            x_data = torch.tensor([[j for j in i[:-1]] for i in data])
            # 一维标签 [batch_size * 1]
            y_data = torch.tensor([i[-1] for i in data]).squeeze_()
           
            # out [BATCH_SIZE, DEPPATH_SIZE * EMBEDDING_SIZE], mean [BATCH_SiZE, LATENT_DIM], log_var [BATCH_SIZE, LATENT_DIM]
            out_vae, mean, log_var, out_pred = cvae.compute_loss(x_data)
#            print (out.shape, mean.shape, log_var.shape)
            
            out_vae = out_vae.squeeze()
            vae_embedding = cvae.embeddings(x_data)
#            print (batch_embedding_y)
#            print('out_vae:\n{0}\nvae_embedding:\n{1}'.format(out_vae, vae_embedding))
            
            reconst_loss, kl_divergence, vae_loss = loss_func(out_vae, vae_embedding, mean, log_var)
#            print ('step: {3} | reconst_loss: {0} | kl_divergence: {1} | total_loss: {2}'.format(reconst_loss, kl_divergence, total_loss, step))
#             nn.CrossEntropyLoss() 自动计算softmax 所以传入的是 [batch * class_count] target [class_count]
            pred_loss = loss_func_pred(out_pred, y_data)
            
#            total_loss = pred_loss + vae_loss
            total_loss = vae_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            loss_list_1.append(vae_loss.item())
            if step % 10 == 0:
                loss_list.append(total_loss.item())
                count += 1
                print ('step: {3} | reconst_loss: {0} | kl_divergence: {1} | total_loss: {2}, pred_loss: {3}'.format(reconst_loss, kl_divergence, total_loss, pred_loss))
            
    # 取最后一个epoch 损失平均值作为 判别elbo
    elbo = 0
    epoch_loop = int(len(torch_dataset) / BATCH_SIZE)
    for i in loss_list_1[-epoch_loop:]:
        elbo += i
    av_elbo = round(elbo / epoch_loop, 2)
    
    # 储存模型
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)
    
    torch.save(cvae, '{0}/cvae.pkl'.format(SAVE_PATH)) # entire net
    
    print ('Save model: {0}/cvae.pkl, elbo: {1}'.format(SAVE_PATH, av_elbo))
    
    # 绘制 loss 曲线
    x1 = [i for i in range(0, count)]
    y1 = loss_list
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, '.-')
    plt.xlabel('steps')
    plt.ylabel('Total loss')
    plt.show()
    
    # filter setp
    print ('filter')

    result_DepPath = []
    test_matrix = torch.from_numpy(test_matrix)
    
    time = 0
    for i in test_matrix:
        time += 1
        if time % 50000 == 0:
            print (time)
#        print (i, i.shape)
        i.unsqueeze_(0)
        out_vae, mean, log_var, out_pred = cvae.compute_loss(i)
        
        out_vae = out_vae.squeeze()
        vae_embedding = cvae.embeddings(i)
        
        out_vae.unsqueeze_(0)
        reconst_loss, kl_divergence, vae_loss = loss_func(out_vae, vae_embedding, mean, log_var)
        
#        pred_loss = loss_func_pred(out_pred, y_data)
        
        total_loss = vae_loss
        
        if total_loss <av_elbo:
            result_DepPath.append(i.data.numpy())
            if len(result_DepPath) == 100:
                print (len(result_DepPath))
    
    # 将符合elbo的路径转化为路径
    id_to_word = {word_to_id[i]:i for i in word_to_id.keys()}
    result_path = [[id_to_word[j] for j in list(i[0]) if j != word_to_id['padding']] for i in result_DepPath]
    
    with open(OUT, 'w') as wf:
        for i in result_path:
            wf.write('start_entity|{0}|end_entity\n'.format('|'.join(i)))
    
if __name__ == '__main__':
    main()