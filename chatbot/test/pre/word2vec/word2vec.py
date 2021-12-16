import collections
import random
import math
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

data_path = '/home/wyundi/Server/Courses/BIA667/Project/Chatbot/data/simple-examples/data'
ptb_train_path = data_path + '/ptb.train.txt'
ptb_valid_path = data_path + '/ptb.valid.txt'
ptb_test_path = data_path + '/ptb.test.txt'

class file_data:

    def __init__(self, data_path):
        self.__raw_dataset = []
        self.__counter = {}
        self.__idx_to_token = []
        self.__token_to_idx = {}
        self.__dataset_idx = []
        self.__subsampled_dataset = []
        self.__num_tokens = 0
        self.__centers = []
        self.__contexts = []
        self.__all_negatives = []

        self.__read_file(data_path)

        # 二次采样，抛弃高频词
        self.__subsampled_dataset = [[tk for tk in st if not self.__discard(tk)] for st in self.__dataset_idx]
        # 提取中心词和背景词
        self.__centers, self.__contexts = self.__get_centers_and_contexts(self.__subsampled_dataset, 5)
        # 负采样
        self.__sampling_weights = [self.__counter[w]**0.75 for w in self.__idx_to_token]
        self.__all_negatives = self.__get_negatives(self.__contexts, self.__sampling_weights, 5)

    def __read_file(self, data_path):
        with open(data_path, 'r') as f:
            self.__lines = f.readlines()
            self.__raw_dataset = [st.split() for st in self.__lines]

            # 记录每个词出现的数量 -> dict
            self.__counter = collections.Counter([tk for st in self.__raw_dataset for tk in st])
            self.__counter = dict(filter(lambda x: x[1] >= 5, self.__counter.items()))

            self.__idx_to_token = [tk for tk, _ in self.__counter.items()]
            self.__token_to_idx = {tk: idx for idx, tk in enumerate(self.__idx_to_token)}

            self.__dataset_idx = [[self.__token_to_idx[tk] for tk in st if tk in self.__idx_to_token] for st in self.__raw_dataset]
            self.__num_tokens = sum([len(st) for st in self.__dataset_idx])

    def __discard(self, idx):
        return random.uniform(0, 1) < 1 - math.sqrt(
                1e-4 / self.__counter[self.__idx_to_token[idx]] * self.__num_tokens)

    def compare_counts(self, token):
        '''
        二次采样前后，token在词库中的数量统计
        the: before=50770, after=2013
        join: before=45, after=45

        print(self.compare_counts('the'))
        print(self.compare_counts('join'))
        '''
        return '# %s: before=%d, after=%d' % (token, sum(
                [st.count(self.__token_to_idx[token]) for st in self.__dataset_idx]), sum(
                [st.count(self.__token_to_idx[token]) for st in self.__subsampled_dataset]))

    def __get_tk(self, dataset):
        return [[self.__idx_to_token[idx] for idx in st] for st in dataset]

    def get_dataset(self):
        return self.__dataset_idx, self.__get_tk(self.__dataset_idx)

    def get_sub_dataset_tk(self):
        return self.__subsampled_dataset, self.__get_tk(self.__subsampled_dataset)

    def __get_centers_and_contexts(self, dataset, max_window_size):
        centers, contexts = [], []
        for st in dataset:
            if len(st) < 2:  # 每个句子至少要有2个词才可能组成一对“中心词-背景词”
                continue
            centers += st
            for center_i in range(len(st)):
                window_size = random.randint(1, max_window_size)
                indices = list(range(max(0, center_i - window_size),
                                    min(len(st), center_i + 1 + window_size)))
                indices.remove(center_i)  # 将中心词排除在背景词之外
                contexts.append([st[idx] for idx in indices])
        return centers, contexts

    def get_center_contexts(self):
        return self.__center, self.__contexts

    def __get_negatives(self, all_contexts, sampling_weights, K):
        all_negatives, neg_candidates, i = [], [], 0
        population = list(range(len(sampling_weights)))
        for contexts in all_contexts:
            negatives = []
            while len(negatives) < len(contexts) * K:
                if i == len(neg_candidates):
                    # 根据每个词的权重（sampling_weights）随机生成k个词的索引作为噪声词。
                    # 为了高效计算，可以将k设得稍大一点
                    i, neg_candidates = 0, random.choices(
                        population, sampling_weights, k=int(1e5))
                neg, i = neg_candidates[i], i + 1
                # 噪声词不能是背景词
                if neg not in set(contexts):
                    negatives.append(neg)
            all_negatives.append(negatives)
        return all_negatives

    def get_negatives(self):
        return self.__all_negatives

    def get_dataset(self):
        return self.__centers, self.__contexts, self.__all_negatives

    def get_idx_to_tk(self):
        return self.__idx_to_token

    def get_tk_to_idx(self):
        return self.__token_to_idx

ptb_train_data = file_data(ptb_train_path)

all_centers, all_contexts, all_negatives = ptb_train_data.get_dataset()
idx_to_token = ptb_train_data.get_idx_to_tk()
token_to_idx = ptb_train_data.get_tk_to_idx()

class word2vec(Dataset):
    def __init__(self, centers, contexts, negatives):
        assert len(centers) == len(contexts) == len(negatives)
        self.centers = centers
        self.contexts = contexts
        self.negatives = negatives

    def __getitem__(self, index):
        return (self.centers[index], self.contexts[index], self.negatives[index])

    def __len__(self):
        return len(self.centers)

def batchify(data):
    """用作DataLoader的参数collate_fn: 输入是个长为batchsize的list, 
    list中的每个元素都是Dataset类调用__getitem__得到的结果
    """
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (torch.tensor(centers).view(-1, 1), torch.tensor(contexts_negatives),
            torch.tensor(masks), torch.tensor(labels))

batch_size = 512
num_workers = 4

dataset = word2vec(all_centers, 
                    all_contexts, 
                    all_negatives)
data_iter = DataLoader(dataset, batch_size, shuffle=True,
                            collate_fn=batchify, 
                            num_workers=num_workers)

'''
for batch in data_iter:
    for name, data in zip(['centers', 'contexts_negatives', 'masks',
                           'labels'], batch):
        print(name, 'shape:', data.shape)
    break

embed = nn.Embedding(num_embeddings=20, embedding_dim=4)
print(embed.weight)
'''

# 前向计算
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred

# 损失函数
class SigmoidBinaryCrossEntropyLoss(nn.Module):
    def __init__(self): # none mean sum
        super(SigmoidBinaryCrossEntropyLoss, self).__init__()
    def forward(self, inputs, targets, mask=None):
        """
        input – Tensor shape: (batch_size, len)
        target – Tensor of the same shape as input
        """
        inputs, targets, mask = inputs.float(), targets.float(), mask.float()
        res = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none", weight=mask)
        return res.mean(dim=1)

loss = SigmoidBinaryCrossEntropyLoss()

# 初始化模型
embed_size = 100
net = nn.Sequential(
    nn.Embedding(num_embeddings=len(idx_to_token), embedding_dim=embed_size),
    nn.Embedding(num_embeddings=len(idx_to_token), embedding_dim=embed_size)
)

# 训练模型
def train(net, lr, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("train on ", device)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    for epoch in range(num_epochs):
        start, l_sum, n = time.time(), 0.0, 0
        for batch in data_iter:
            center, context_negative, mask, label = [d.to(device) for d in batch]

            pred = skip_gram(center, context_negative, net[0], net[1])

            # 使用掩码变量mask来避免填充项对损失函数计算的影响
            l = (loss(pred.view(label.shape), label, mask) *
                 mask.shape[1] / mask.float().sum(dim=1)).mean() # 一个batch的平均loss
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_sum += l.cpu().item()
            n += 1
        print('epoch %d, loss %.2f, time %.2fs'
              % (epoch + 1, l_sum / n, time.time() - start))

train(net, 0.01, 10)

def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data
    x = W[token_to_idx[query_token]]
    # print(x)

    # 添加的1e-9是为了数值稳定性
    cos = torch.matmul(W, x) / (torch.sum(W * W, dim=1) * torch.sum(x * x) + 1e-9).sqrt()
    _, topk = torch.topk(cos, k=k+1)
    topk = topk.cpu().numpy()
    for i in topk[1:]:  # 除去输入词
        print('cosine sim=%.3f: %s' % (cos[i], (idx_to_token[i])))

get_similar_tokens('chip', 3, net[0])