# This file contains Transformer network
# Most of the code is copied from http://nlp.seas.harvard.edu/2018/04/03/attention.html

# The cfg name correspondance:
# N=num_layers
# d_model=input_encoding_size
# d_ff=rnn_size
# h is always 8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils as utils

import copy
import math
import numpy as np

from CaptionModel import CaptionModel
from AttModel_transf import sort_pack_padded_sequence, pad_unsort_packed_sequence, pack_wrapper, AttModel

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, node_logits, rel_scores, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, node_logits, rel_scores, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, node_logits, rel_scores, src_mask):
        "src_embed here seems nothing to do, it is lambda x:x"
        return self.encoder(self.src_embed(src), node_logits, rel_scores, src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, node_logits, rel_scores, mask):
        "Pass the input (and mask) through each layer in turn."
        p_node = F.softmax(torch.sum(node_logits, dim=2),dim=1)
        A_rel = torch.sum(rel_scores, dim=3)
        for layer in self.layers:
            x, p_node, A_rel = layer(x, p_node, node_logits, A_rel, rel_scores, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, node_attn, rel_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.node_attn = node_attn
        self.rel_attn = rel_attn
        self.feed_forward = feed_forward
        self.normlayer = clones(LayerNorm(size),3)
        self.dropout = nn.Dropout(dropout)
        self.size = size
        self.sublayer = SublayerConnection(size, dropout)

    def forward(self, x, x_node, node_logits, A_rel, rel_scores, mask):
        "Follow Figure 1 (left) for connections."
        x_org = self.self_attn(x, x, x, mask)
        x_node, x_node_vec = self.node_attn(x_node, node_logits)
        A_rel, rel_vec = self.rel_attn(A_rel, rel_scores, x_org, mask)
        y = x + self.dropout(self.normlayer[0](x_org)) + self.dropout(self.normlayer[1](x_node_vec)) + self.dropout(self.normlayer[2](rel_vec))
        return self.sublayer(y, self.feed_forward), x_node, A_rel

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
#         print('query shape is {}'.format(query.shape))

        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
    
class NodeSemanticAttention(nn.Module):
    def __init__(self, obj_glove, d_model, dropout=0.1):
        super(NodeSemanticAttention, self).__init__()
        self.obj_glove = obj_glove
        self.glove_embed = nn.Linear(obj_glove.shape[1], d_model)
        self.linear = nn.Linear(d_model, d_model)
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.linear_emb = nn.Linear(d_model, 1)
        self.d_model = d_model
        
    def forward(self, p_node, node_logits):
        num_box_cls = node_logits.shape[2]
        bs = node_logits.shape[0]
        num_box = node_logits.shape[1]
        
        ## get semantic reps by word embeddings
        box_glove_emb = torch.mm(node_logits.view(-1,num_box_cls),self.glove_embed(self.obj_glove)).view(bs,num_box,-1)
        
        ## comput ps = softmax(wi * att_feats) --> batchsize * 36(number of nodes)
        pn_box_feats = self.linear(box_glove_emb)           # batch * num_nodes * rnn_size
        ps = p_node + self.linear_emb(torch.tanh(pn_box_feats)).squeeze(2) * p_node
        x_node = F.softmax(ps,dim=1)
        
        x_node_vec = pn_box_feats * ps.unsqueeze(2).expand(bs, num_box, self.d_model)
        return x_node, x_node_vec
    
class RelationAttention(nn.Module):
    def __init__(self, src_attn, vocab_glove, rel_dic, top_num, d_model, dropout=0.1):
        super(RelationAttention, self).__init__()
        self.rel_dic, self.vocab_glove, self.d_model, self.top_num = rel_dic, vocab_glove, d_model, top_num
        
        self.rel_embed = nn.Linear(vocab_glove.shape[1], d_model)
        self.dic_embed = nn.Linear(d_model*3, d_model)
        
        self.src_attn = src_attn
        self.norm = LayerNorm(d_model)
        
        self.linear = nn.Linear(d_model, d_model)
        self.feat_emb = nn.Linear(d_model*2, d_model)
        
    def forward(self, A_rel, rel_scores, att_feats, att_mask):
        bs = rel_scores.shape[0]
        num_node = rel_scores.shape[1]
        num_rel_cls = rel_scores.shape[3]       
        
        ## get glove vec of rel_dic
        rel_glove = self.dic_embed(self.rel_embed(self.vocab_glove[self.rel_dic]).view(num_rel_cls, -1))        
        ## get rel vector
        rel_vecs =  torch.mm(rel_scores.view(-1, num_rel_cls), rel_glove).view(bs,num_node*num_node,-1) ## bs * num_nodes^2 * rnn_size
                 
        ## get topk relation representations
        topk, top_idx = torch.topk(A_rel.view(bs, -1), self.top_num, dim=1, sorted=False)
        topk = F.softmax(topk, dim=1)
        idx = top_idx.unsqueeze(-1).expand(bs, self.top_num, self.d_model)
        pe_rel_topk = torch.gather(rel_vecs, dim=1, index=idx) * topk.unsqueeze(-1)  ## bs * top_num * rnn_size  

        ##change top_num * rnn_size to num_node * rnn_size    
        rel_reps = self.norm(self.src_attn(att_feats, pe_rel_topk, pe_rel_topk))
        
        ## refresh  A
        pe_rel_reps = self.linear(rel_reps)                             # batch * rnn_size
        node_pre = self.feat_emb(torch.cat([pe_rel_reps, att_feats],dim=2))
        A_added = torch.matmul(node_pre, node_pre.transpose(1, 2))
        A_new = A_added + A_rel
        
        return A_new, pe_rel_reps

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, vocab_glove, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear_emb = nn.Linear(vocab_glove.shape[1], d_model)
        self.vocab_glove = vocab_glove
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, seqs):
        x = self.linear_emb(self.vocab_glove[seqs])
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    
class LSTMEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, vocab_glove, d_model, dropout, max_len=5000):
        super(LSTMEncoding, self).__init__()
        self.d_model = d_model
        self.lstm = nn.LSTMCell(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.linear_emb = nn.Linear(vocab_glove.shape[1], d_model)
        self.vocab_glove = vocab_glove
        
    def forward(self, seqs):
        x = self.linear_emb(self.vocab_glove[seqs])
        bsz = x.size(0)
        state = self.init_hidden(bsz)
        output = x.new_zeros((bsz, x.size(1), x.size(2)))
        for i in range(x.size(1)):
            h, c = self.lstm(x[:,i,:], (state[0][0], state[1][0]))
            output[:,i,:] = self.dropout(h)
            state = (torch.stack([h]), torch.stack([c]))
            
        return output
    
    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(1, bsz, self.d_model),
                weight.new_zeros(1, bsz, self.d_model))

class TransformerModel(AttModel):

    def make_model(self, src_vocab, tgt_vocab, N=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1):
        "Helper: Construct a model from hyperparameters."
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(self.opt.vocab_glove, d_model, dropout)
#         position = LSTMEncoding(self.opt.vocab_glove, d_model, dropout)
        nodeattn = NodeSemanticAttention(self.opt.obj_glove, d_model, dropout)
        rel_attn = RelationAttention(c(attn), self.opt.vocab_glove, self.opt.rel_dic, self.opt.top_num, d_model, dropout)
        model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(nodeattn), c(rel_attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                                 c(ff), dropout), N),
            lambda x:x, # nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
            c(position),
            Generator(d_model, tgt_vocab))
        
        # This was important from their code. 
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, opt):
        super(TransformerModel, self).__init__(opt)
        self.opt = opt
        # self.config = yaml.load(open(opt.config_file))
        # d_model = self.input_encoding_size # 512

        delattr(self, 'att_embed')
        self.att_embed = nn.Sequential(*(
                                    ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ())+
                                    (nn.Linear(self.att_feat_size, self.input_encoding_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))+
                                    ((nn.BatchNorm1d(self.input_encoding_size),) if self.use_bn==2 else ())))
        
        delattr(self, 'embed')
        self.embed = lambda x : x
        delattr(self, 'fc_embed')
        self.fc_embed = lambda x : x
        delattr(self, 'logit')
        del self.ctx2att

        tgt_vocab = self.vocab_size + 1
#         print('------tgt vocab is {}------'.format(tgt_vocab))
        self.model = self.make_model(0, tgt_vocab,
            N=opt.num_layers,
            d_model=opt.input_encoding_size,
            d_ff=opt.rnn_size)

    def logit(self, x): # unsafe way
        return self.model.generator.proj(x)

    def init_hidden(self, bsz):
        return []

    def _prepare_feature(self, fc_feats, att_feats, node_logits, rel_scores, att_masks):

        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks)
        memory = self.model.encode(att_feats, node_logits, rel_scores, att_masks)

        return fc_feats[...,:1], att_feats[...,:1], memory, att_masks

    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)

        if seq is not None: 
            # crop the last one
            seq = seq[:,:-1]
            seq_mask = (seq.data > 0).type(torch.uint8)
            seq_mask[:,0] += 1
            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None

        return att_feats, seq, att_masks, seq_mask

    def _forward(self, fc_feats, att_feats, seq, box_logits, att_relscore, att_masks=None):
#         print("---------------transformer start-----------------")
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq)

        out = self.model(att_feats, seq, box_logits, att_relscore, att_masks, seq_mask)
#         print(out.shape)

        outputs = self.model.generator(out)
#         print(outputs.shape)
        return outputs
        # return torch.cat([_.unsqueeze(1) for _ in outputs], 1)

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):
        """
        state = [ys.unsqueeze(0)]
        """
        if len(state) == 0:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
        out = self.model.decode(memory, mask, 
                               ys, 
                               subsequent_mask(ys.size(1))
                                        .to(memory.device))
        return out[:, -1], [ys.unsqueeze(0)]