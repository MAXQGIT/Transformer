import torch
from review import Transformers
import torch.utils.data as Data

model = Transformers()
model.load_state_dict(torch.load('model.pth'))

# 德语和英语的单词要分开建立词库
# Padding should be Zero
src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4, 'cola': 5}
# 枚举
src_idx2word = {i: w for i, w in enumerate(src_vocab)}
src_vocab_size = len(src_vocab)

tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6}
idx2word = {i: w for i, w in enumerate(tgt_vocab)}
tgt_vocab_size = len(tgt_vocab)

# 预测阶段

start_symbol = tgt_vocab["S"]


def greedy_decoder(model, enc_inputs, start_symbol):
    enc_outputs = model.encoder(enc_inputs)
    dec_input = torch.zeros(1, 0).type_as(enc_inputs.data)
    terminal = False
    next_symbol = start_symbol
    while not terminal:
        # 预测阶段：dec_input序列会一点点变长（每次添加一个新预测出来的单词）
        dec_input = torch.cat([dec_input, torch.tensor([[next_symbol]], dtype=enc_inputs.dtype)],
                              -1)
        dec_outputs = model.decoder(dec_input, enc_inputs, enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        # 增量更新（我们希望重复单词预测结果是一样的）
        # 我们在预测是会选择性忽略重复的预测的词，只摘取最新预测的单词拼接到输入序列中
        next_word = prob.data[-1]  # 拿出当前预测的单词(数字)。我们用x'_t对应的输出z_t去预测下一个单词的概率，不用z_1,z_2..z_{t-1}
        next_symbol = next_word
        if next_symbol == tgt_vocab["E"]:
            terminal = True

    greedy_dec_predict = dec_input[:, 1:]
    return greedy_dec_predict


enc_inputs = torch.LongTensor([1, 2, 3, 4, 0])
print(enc_inputs.size())

greedy_dec_predict = greedy_decoder(model, enc_inputs.view(1, -1), start_symbol=tgt_vocab["S"])
print(enc_inputs, '->', greedy_dec_predict.squeeze())

print([src_idx2word[t.item()] for t in enc_inputs.data])
print([idx2word[n.item()] for n in greedy_dec_predict.squeeze()])
