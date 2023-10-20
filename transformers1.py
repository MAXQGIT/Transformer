import torch
from torch import nn

# -------------------------------------------------- #
# （1）muti_head_attention
# -------------------------------------------------- #
'''
embed_size: 每个单词用多少长度的向量来表示
heads: 多头注意力的heads个数
'''


class selfattention(nn.Module):
    def __init__(self, embed_size, heads):
        super(selfattention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        # 每个head的处理的特征个数
        self.head_dim = embed_size // heads

        # 如果不能整除就报错
        assert (self.head_dim * self.heads == self.embed_size), 'embed_size should be divided by heads'

        # 三个全连接分别计算qkv
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

        # 输出层
        self.fc_out = nn.Linear(self.head_dim * self.heads, embed_size)

    # 前向传播 qkv.shape==[b,seq_len,embed_size]
    def forward(self, values, keys, query, mask):
        N = query.shape[0]  # batch
        # 获取每个句子有多少个单词
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # 维度调整 [b,seq_len,embed_size] ==> [b,seq_len,heads,head_dim]
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        # 对原始输入数据计算q、k、v
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # 爱因斯坦简记法，用于张量矩阵运算，q和k的转置矩阵相乘
        # queries.shape = [N, query_len, self.heads, self.head_dim]
        # keys.shape = [N, keys_len, self.heads, self.head_dim]
        # energy.shape = [N, heads, query_len, keys_len]
        energy = torch.einsum('nqhd, nkhd -> nhqk', [queries, keys])

        # 是否使用mask遮挡t时刻以后的所有q、k
        if mask is not None:
            # 将mask中所有为0的位置的元素，在energy中对应位置都置为 －1*10^10
            energy = energy.masked_fill(mask == 0, torch.tensor(-1e10))

        # 根据公式计算attention, 在最后一个维度上计算softmax
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        # 爱因斯坦简记法矩阵元素，其中query_len == keys_len == value_len
        # attention.shape = [N, heads, query_len, keys_len]
        # values.shape = [N, value_len, heads, head_dim]
        # out.shape = [N, query_len, heads, head_dim]
        out = torch.einsum('nhql, nlhd -> nqhd', [attention, values])

        # 维度调整 [N, query_len, heads, head_dim] ==> [N, query_len, heads*head_dim]
        out = out.reshape(N, query_len, self.heads * self.head_dim)

        # 全连接，shape不变
        out = self.fc_out(out)
        return out


# -------------------------------------------------- #
# （2）multi_head_attention + FFN
# -------------------------------------------------- #
'''
embed_size: wordembedding之后, 每个单词用多少长度的向量来表示
heads: 多头注意力的heas个数
drop: 杀死神经元的概率
forward_expansion:  在FFN中第一个全连接上升特征数的倍数
'''


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()

        # 实例化自注意力模块
        self.attention = selfattention(embed_size, heads)

        # muti_head之后的layernorm
        self.norm1 = nn.LayerNorm(embed_size)
        # FFN之后的layernorm
        self.norm2 = nn.LayerNorm(embed_size)

        # 构建FFN前馈型神经网络
        self.feed_forward = nn.Sequential(
            # 第一个全连接层上升特征个数
            nn.Linear(embed_size, embed_size * forward_expansion),
            # relu激活
            nn.ReLU(),
            # 第二个全连接下降特征个数
            nn.Linear(embed_size * forward_expansion, embed_size)
        )

        # dropout层随机杀死神经元
        self.dropout = nn.Dropout(dropout)

    # 前向传播, qkv.shape==[b,seq_len,embed_size]
    def forward(self, value, key, query, mask):
        # 计算muti_head_attention
        attention = self.attention(value, key, query, mask)
        # 输入和输出做残差连接
        x = query + attention
        # layernorm标准化
        x = self.norm1(x)
        # dropout
        x = self.dropout(x)

        # FFN
        ffn = self.feed_forward(x)
        # 残差连接输入和输出
        forward = ffn + x
        # layernorm + dropout
        out = self.dropout(self.norm2(forward))

        return out


# -------------------------------------------------- #
# （3）encoder
# -------------------------------------------------- #
'''
src_vocab_size: 一共有多少个单词
num_layers: 堆叠多少层TransformerBlock
device: GPU or CPU
max_len: 最长的一个句子有多少个单词
embed_size: wordembedding之后, 每个单词用多少长度的向量来表示
heads: 多头注意力的heas个数
drop: 在muti_head_atten和FFN之后的dropout层杀死神经元的概率
forward_expansion:  在FFN中第一个全连接上升特征数的倍数
'''


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, num_layers, device, max_len,
                 embed_size, heads, dropout, forward_expansion):
        super(Encoder, self).__init__()

        self.embed_size = embed_size
        self.device = device

        # wordembedding 将每个单词用长度为多少的向量来表示
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        # 对每一个单词的位置编码
        self.position_embedding = nn.Embedding(max_len, embed_size)

        self.dropout = nn.Dropout(dropout)

        # 将多个TransformerBlock保存在列表中
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, heads, dropout, forward_expansion)
             for _ in range(num_layers)]
        )

    # 前向传播x.shape=[batch, seq_len]
    def forward(self, x, mask):
        # 获取输入句子的shape
        N, seq_len = x.shape

        # 为每个单词构造位置信息, 并搬运到GPU上
        position = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)

        # 将输入的句子经过wordembedding和位置编码后相加 [batch, seq_len, embed_size]
        out = self.word_embedding(x) + self.position_embedding(position)
        # dropout层
        out = self.dropout(out)

        # 堆叠多个TransformerBlock层
        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out


# -------------------------------------------------- #
# （4）decoder_block
# -------------------------------------------------- #
'''
embed_size: wordembedding之后, 每个单词用多少长度的向量来表示
heads: 多头注意力的heas个数
drop: 在muti_head_atten和FFN之后的dropout层杀死神经元的概率
forward_expansion:  在FFN中第一个全连接上升特征数的倍数
'''


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(DecoderBlock, self).__init__()

        # 实例化muti_head_attention
        self.attention = selfattention(embed_size, heads)
        # 实例化TransformerBlock
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)

        # 第一个muti_head_atten之后的LN和Dropout
        self.norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    # 前向传播
    def forward(self, x, value, key, src_mask, trg_mask):
        # 对output计算self_attention
        attention = self.attention(x, x, x, trg_mask)
        # 残差连接
        query = self.dropout(self.norm(attention + x))

        # 将encoder部分的k、v和decoder部分的q做TransformerBlock
        out = self.transformer_block(value, key, query, src_mask)
        return out


# -------------------------------------------------- #
# （5）decoder
# -------------------------------------------------- #
'''
trg_vocab_size: 目标句子的长度
num_layers: 堆叠多少个decoder_block
max_len: 目标句子中最长的句子有几个单词
device: GPU or CPU
embed_size: wordembedding之后, 每个单词用多少长度的向量来表示
heads: 多头注意力的heas个数
drop: 在muti_head_atten和FFN之后的dropout层杀死神经元的概率
forward_expansion:  在FFN中第一个全连接上升特征数的倍数
'''


class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, num_layers, device, max_len,
                 embed_size, heads, forward_expansion, dropout):
        super(Decoder, self).__init__()

        self.device = device

        # trg_vocab_size代表目标句子的单词总数，embed_size代表每个单词用多长的向量来表示
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        # 位置编码，max_len代表目标句子中最长有几个单词
        self.position_embeddimg = nn.Embedding(max_len, embed_size)

        # 堆叠多个decoder_block
        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size, heads, forward_expansion, dropout)
             for _ in range(num_layers)]
        )

        # 输出层
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    # 前向传播
    def forward(self, x, enc_out, src_mask, trg_mask):
        # 获取decoder部分输入的shape=[batch, seq_len]
        N, seq_len = x.shape

        # 位置编码
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)

        # word_embedding和位置编码后的结果相加
        x = self.word_embedding(x) + self.position_embeddimg(x)
        x = self.dropout(x)

        # 堆叠多个DecoderBlock, 其中它的key和value是用的encoder的输出 [batch, seq_len, embed_size]
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        # 输出层
        out = self.fc_out(x)
        return out


# -------------------------------------------------- #
# （6）模型构建
# -------------------------------------------------- #
class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 src_pad_idx,
                 trg_pad_idx,
                 embed_size=512,
                 num_layers=6,
                 forward_expansion=4,
                 heads=8,
                 dropout=0,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 max_len=100):
        super(Transformer, self).__init__()

        self.encoder = Encoder(src_vocab_size,
                               num_layers,
                               device,
                               max_len,
                               embed_size,
                               heads,
                               dropout,
                               forward_expansion)

        self.decoder = Decoder(trg_vocab_size,
                               num_layers,
                               device,
                               max_len,
                               embed_size,
                               heads,
                               forward_expansion,
                               dropout)

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    # 构造mask
    def make_src_mask(self, src):
        # [N,src_len]==>[N,1,1,src_len]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        # 获取目标句子的shape
        N, trg_len = trg.shape
        # 构造mask  [trg_len, trg_len]==>[N, 1, trg_len, trg_len]
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
        return trg_mask.to(self.device)

    # 前向传播
    def forward(self, src, trg):
        # 对输入句子构造mask
        src_mask = self.make_src_mask(src)
        # 对目标句子构造mask
        trg_mask = self.make_trg_mask(trg)

        # encoder
        enc_src = self.encoder(src, src_mask)
        # decoder
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out


# -------------------------------------------------- #
# （7）模型测试
# -------------------------------------------------- #
if __name__ == '__main__':
    # 电脑上有GPU就调用它，没有就用CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 输入
    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
    # 目标
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    src_vocab_size = 10  # 输入句子的长度
    trg_vocab_size = 10  # 目标句子的长度

    src_pad_idx = 0  # 对输入句子中的0值做mask
    trg_pad_idx = 0

    # 接收模型
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx)
    model = model.to(device)

    # 前向传播，参数：输入句子和目标句子
    out = model(x, trg[:, :-1])  # 预测最后一个句子

    print(out.shape)
    # torch.Size([2, 7, 10])