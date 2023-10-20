import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']
src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}
src_vocab_size = len(src_vocab)
tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6}
tgt_dict = {i: w for i, w in enumerate(tgt_vocab)}
tgt_vocab_size = len(tgt_vocab)

src_len = 5
tgt_len = 5
eopchs = 20
d_model = 512
layers = 6
d_ff = 2048
heads = 8
d_k = d_v = 64


def mask_batch(sentences):
    input_batch = [[src_vocab[i] for i in sentences[0].split()]]
    output_batch = [[tgt_vocab[i] for i in sentences[1].split()]]
    target_batch = [[tgt_vocab[i] for i in sentences[2].split()]]
    return Variable(torch.LongTensor(input_batch)), Variable(torch.LongTensor(output_batch)), Variable(
        torch.LongTensor(target_batch))


class Transformers(nn.Module):
    def __init__(self):
        super(Transformers, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, enc_inputs, dec_inputs):
        enc_outputs = self.encoder(enc_inputs)
        dec_outputs = self.decoder(dec_inputs,enc_inputs,enc_outputs)
        dec_outputs = self.projection(dec_outputs)
        return dec_outputs.view(-1,dec_outputs.size(-1))


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.pos_embed = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(10, d_model))
        self.layers = nn.ModuleList([Encoderlayer() for _ in range(layers)])

    def forward(self, enc_inputs):
        enc_outouts = self.src_embed(enc_inputs) + self.pos_embed(torch.LongTensor([[i+1 for i in range(enc_inputs.size(1))]]))
        enc_self_attn_mask = get_att_pad_mask(enc_inputs, enc_inputs)
        for layer in self.layers:
            enc_outouts = layer(enc_outouts, enc_self_attn_mask)

        return enc_outouts


class Encoderlayer(nn.Module):
    def __init__(self):
        super(Encoderlayer, self).__init__()
        self.enc_self_att = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_att(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, heads * d_k)
        self.W_K = nn.Linear(d_model, heads * d_k)
        self.W_V = nn.Linear(d_model, heads * d_v)

    def forward(self, q, k, v, attn_mask):
        batch_size = q.size(0)
        residul = q

        q_s = self.W_Q(q).view(batch_size, -1, heads, d_k).transpose(1, 2)
        q_k = self.W_K(k).view(batch_size, -1, heads, d_k).transpose(1, 2)
        q_v = self.W_V(v).view(batch_size, -1, heads, d_v).transpose(1, 2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, heads, 1, 1)
        context, atten = ScaledDotProductAttention()(q_s, q_k, q_v, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, heads * d_v)
        output = nn.Linear(heads * d_v, d_model)(context)
        return nn.LayerNorm(d_model)(residul + output), atten


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attention = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attention, V)
        return context, attention


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        residual = inputs
        output = self.conv1(inputs.transpose(1, 2))
        output = self.relu(output)
        output = self.conv2(output).transpose(1, 2)
        return nn.LayerNorm(d_model)(residual + output)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.dec_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(10, d_model))

        self.layers= nn.ModuleList([Decoderlayer() for _ in range(layers)])

    def forward(self,outputs,inputs,enc_outputs):
        dec_outputs = self.dec_emb(outputs) + self.pos_emb(torch.LongTensor([[i+1 for i in range(outputs.size(1))]]))
        dec_self_attn_pad_mask = get_att_pad_mask(outputs, outputs)
        dec_subsentence_mask = get_attn_subsequent_mask(outputs)
        dec_self_atten_mask = torch.gt((dec_self_attn_pad_mask+dec_subsentence_mask),0)
        dec_enc_self_mask = get_att_pad_mask(outputs,inputs)
        for layer in self.layers:
            dec_outputs = layer(enc_outputs,dec_outputs,dec_self_atten_mask,dec_enc_self_mask)
        return dec_outputs

class Decoderlayer(nn.Module):
    def __init__(self):
        super(Decoderlayer, self).__init__()
        self.dec_self_atten = MultiHeadAttention()
        self.dec_enc_atten = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self,enc_outouts,dec_inputs,dec_self_atten_mask,dec_enc_self_mask):
        dec_outputs,atten = self.dec_self_atten(dec_inputs,dec_inputs,dec_inputs,dec_self_atten_mask)
        dec_outputs,atten =self.dec_enc_atten(dec_outputs,enc_outouts,enc_outouts,dec_enc_self_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs


def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.FloatTensor(sinusoid_table)


def get_att_pad_mask(seq_k, seq_v):
    batch_size, len_q = seq_k.size()
    batch_size, len_k = seq_v.size()
    pad_attn_mask = seq_v.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)


def get_attn_subsequent_mask(dec_inputs):
    attn_shape = [dec_inputs.size(0), dec_inputs.size(1), dec_inputs.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape))
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask


if __name__=='__main__':

    input_data, outout_data, tar_data = mask_batch(sentences)
    model = Transformers()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(input_data,outout_data)
        print(outputs)
        loss = criterion(outputs,tar_data.contiguous().view(-1))
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()


    torch.save(model.state_dict(),'model.pth')


    # enc_inputs = torch.LongTensor([1, 2, 3, 4, 0])

    # dec_outputs = model.decoder(enc_outputs,dec_input)
