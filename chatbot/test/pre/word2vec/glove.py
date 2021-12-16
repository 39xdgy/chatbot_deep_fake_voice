import torch
import torchtext.vocab as vocab

# 使用预训练的词向量
print(vocab.pretrained_aliases.keys())

print([key for key in vocab.pretrained_aliases.keys()
        if "glove" in key])

cache_dir = "/home/wyundi/Server/Courses/BIA667/Project/Chatbot/data/glove"
# glove = vocab.pretrained_aliases["glove.twitter.27B.200d"](cache=cache_dir)
glove = vocab.GloVe(name='42B', dim=300, cache=cache_dir) # 与上面等价

print(len(glove.stoi))
print(glove.stoi['beautiful'], glove.itos[3366]) # (3366, 'beautiful')

# 求近义词
def knn(W, x, k):
    # 添加的1e-9是为了数值稳定性
    cos = torch.matmul(W, x.view((-1,))) / (
        (torch.sum(W * W, dim=1) + 1e-9).sqrt() * torch.sum(x * x).sqrt())
    _, topk = torch.topk(cos, k=k)
    topk = topk.cpu().numpy()
    return topk, [cos[i].item() for i in topk]

def get_similar_tokens(query_token, k, embed):
    topk, cos = knn(embed.vectors,
                    embed.vectors[embed.stoi[query_token]], k+1)
    for i, c in zip(topk[1:], cos[1:]):  # 除去输入词
        print('cosine sim=%.3f: %s' % (c, (embed.itos[i])))

get_similar_tokens('manager', 3, glove)

# 求类比词
def get_analogy(token_a, token_b, token_c, embed):
    vecs = [embed.vectors[embed.stoi[t]] 
                for t in [token_a, token_b, token_c]]
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = knn(embed.vectors, x, 1)
    return embed.itos[topk[0]]

print(get_analogy('man', 'woman', 'son', glove)) # 'daughter'
print(get_analogy('beijing', 'china', 'tokyo', glove)) # 'japan'
print(get_analogy('do', 'did', 'go', glove)) # 'went'