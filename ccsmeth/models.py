#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import torch
import torch.nn as nn

from .utils.constants_torch import use_cuda
from .utils.attention import Attention

from .utils.process_utils import MAX_KINETICS, MAX_PASSES, MAX_SN, MAX_MAP, MAX_KINETICS_STD
from .utils.process_utils import NEMBED_KINETICS, NEMBED_PASSES, NEMBED_SN, NEMBED_MAP, NEMBED_KINETICS_STD

import math  # TODO: is it suitable?

class ModelAttRNN(nn.Module):
    def __init__(self, seq_len=21, num_layers=3, num_classes=2,
                 dropout_rate=0.5, hidden_size=256,
                 vocab_size=16, embedding_size=4,
                 is_npass=True, is_sn=False, is_map=False, is_stds=False, 
                 model_type="attbigru2s",
                 device=0):
        super(ModelAttRNN, self).__init__()
        self.model_type = model_type
        self.device = device

        self.seq_len = seq_len
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        self.embed = nn.Embedding(vocab_size, embedding_size)  # for dna/rna base

        self.is_stds = is_stds
        self.is_npass = is_npass
        self.is_sn = is_sn
        self.is_map = is_map
        self.feas_ccs = 2
        if self.is_stds:
            self.feas_ccs += 2
        if self.is_npass:
            self.feas_ccs += 1
        if self.is_sn:
            self.feas_ccs += 1
        if self.is_map:
            self.feas_ccs += 1
        if self.model_type == "attbilstm2s":
            self.rnn_cell = "lstm"
            self.rnn = nn.LSTM(embedding_size + self.feas_ccs, self.hidden_size, self.num_layers,
                               dropout=dropout_rate, batch_first=True, bidirectional=True)
            # self.rnn2 = nn.LSTM(embedding_size + self.feas_ccs, self.hidden_size, self.num_layers,
            #                     dropout=dropout_rate, batch_first=True, bidirectional=True)
        elif self.model_type == "attbigru2s":
            self.rnn_cell = "gru"
            self.rnn = nn.GRU(embedding_size + self.feas_ccs, self.hidden_size, self.num_layers,
                              dropout=dropout_rate, batch_first=True, bidirectional=True)
            # self.rnn2 = nn.GRU(embedding_size + self.feas_ccs, self.hidden_size, self.num_layers,
            #                    dropout=dropout_rate, batch_first=True, bidirectional=True)
        else:
            raise ValueError("--model_type not set right!")

        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(self.hidden_size * 2 * 2, self.num_classes)  # 2 for bidirection, another 2 for 2 strands

        self._att3 = Attention(self.hidden_size * 2, self.hidden_size * 2, self.hidden_size)
        # self._att3_2 = Attention(self.hidden_size * 2, self.hidden_size * 2, self.hidden_size)

        self.softmax = nn.Softmax(1)

    def get_model_type(self):
        return self.model_type

    def init_hidden(self, batch_size, num_layers, hidden_size):
        # Set initial states
        h0 = torch.randn(num_layers * 2, batch_size, hidden_size, requires_grad=True)
        if use_cuda:
            h0 = h0.cuda(self.device)
        if self.rnn_cell == "lstm":
            c0 = torch.randn(num_layers * 2, batch_size, hidden_size, requires_grad=True)
            if use_cuda:
                c0 = c0.cuda(self.device)
            return h0, c0
        return h0

    def forward(self, kmer, kpass, ipd_means, ipd_stds, pw_means, pw_stds, sns, maps,
                kmer2, kpass2, ipd_means2, ipd_stds2, pw_means2, pw_stds2, sns2, maps2):
        kmer_embed = self.embed(kmer.long())

        ipd_means = torch.reshape(ipd_means, (-1, self.seq_len, 1)).float()
        pw_means = torch.reshape(pw_means, (-1, self.seq_len, 1)).float()
        kmer_embed2 = self.embed(kmer2.long())
        ipd_means2 = torch.reshape(ipd_means2, (-1, self.seq_len, 1)).float()
        pw_means2 = torch.reshape(pw_means2, (-1, self.seq_len, 1)).float()

        out1 = torch.cat((kmer_embed, ipd_means, pw_means), 2)  # (N, L, C)
        out2 = torch.cat((kmer_embed2, ipd_means2, pw_means2), 2)  # (N, L, C)

        if self.is_npass:
            kpass = torch.reshape(kpass, (-1, self.seq_len, 1)).float()
            out1 = torch.cat((out1, kpass), 2)  # (N, L, C)
            kpass2 = torch.reshape(kpass2, (-1, self.seq_len, 1)).float()
            out2 = torch.cat((out2, kpass2), 2)  # (N, L, C)
        if self.is_stds:
            ipd_stds = torch.reshape(ipd_stds, (-1, self.seq_len, 1)).float()
            pw_stds = torch.reshape(pw_stds, (-1, self.seq_len, 1)).float()
            out1 = torch.cat((out1, ipd_stds, pw_stds), 2)  # (N, L, C)
            ipd_stds2 = torch.reshape(ipd_stds2, (-1, self.seq_len, 1)).float()
            pw_stds2 = torch.reshape(pw_stds2, (-1, self.seq_len, 1)).float()
            out2 = torch.cat((out2, ipd_stds2, pw_stds2), 2)  # (N, L, C)
        if self.is_sn:
            sns = torch.reshape(sns, (-1, self.seq_len, 1)).float()
            out1 = torch.cat((out1, sns), 2)  # (N, L, C)
            sns2 = torch.reshape(sns2, (-1, self.seq_len, 1)).float()
            out2 = torch.cat((out2, sns2), 2)  # (N, L, C)
        if self.is_map:
            maps = torch.reshape(maps, (-1, self.seq_len, 1)).float()
            out1 = torch.cat((out1, maps), 2)  # (N, L, C)
            maps2 = torch.reshape(maps2, (-1, self.seq_len, 1)).float()
            out2 = torch.cat((out2, maps2), 2)  # (N, L, C)

        out1, n_states1 = self.rnn(out1, self.init_hidden(out1.size(0),
                                                          self.num_layers,
                                                          self.hidden_size))  # (N, L, nhid*2)
        # out2, n_states2 = self.rnn2(out2, self.init_hidden(out2.size(0),
        #                                                    self.num_layers,
        #                                                    self.hidden_size))  # (N, L, nhid*2)
        out2, n_states2 = self.rnn(out2, self.init_hidden(out2.size(0),
                                                          self.num_layers,
                                                          self.hidden_size))  # (N, L, nhid*2)

        # attention_net3 ======
        # h_n: (num_layer * 2, N, nhid), h_0, c_0 -> h_n, c_n not affected by batch_first
        # h_n (last layer) = out[:, -1, :self.hidden_size] concats out1[:, 0, self.hidden_size:]
        h_n1 = n_states1[0] if self.rnn_cell == "lstm" else n_states1
        h_n1 = h_n1.reshape(self.num_layers, 2, -1, self.hidden_size)[-1]  # last layer (2, N, nhid)
        h_n1 = h_n1.transpose(0, 1).reshape(-1, 1, 2 * self.hidden_size)
        out1, att_weights1 = self._att3(h_n1, out1)

        h_n2 = n_states2[0] if self.rnn_cell == "lstm" else n_states2
        h_n2 = h_n2.reshape(self.num_layers, 2, -1, self.hidden_size)[-1]  # last layer (2, N, nhid)
        h_n2 = h_n2.transpose(0, 1).reshape(-1, 1, 2 * self.hidden_size)
        # out2, att_weights2 = self._att3_2(h_n2, out2)
        out2, att_weights2 = self._att3(h_n2, out2)

        out = torch.cat((out1, out2), 1)

        out = self.dropout1(out)
        out = self.fc1(out)

        return out, self.softmax(out)


# Transformer Encoder ===============================================================
# from pytorch/examples
# from https://nlp.seas.harvard.edu/annotated-transformer/#positional-encoding
# from https://medium.com/@hunter-j-phillips/positional-encoding-7a93db4109e6
# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=100).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        pe = pe.unsqueeze(0)  # batch_first version
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        # x = x + self.pe[:x.size(0), :]
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)  # batch_first version
        return self.dropout(x)


class ModelTransEnc(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""
    def __init__(self, seq_len=21, num_layers=6, num_classes=2,
                 dropout_rate=0.5, d_model=256, nhead=4, dim_ff=512,
                 nvocab=16, nembed=4, 
                 is_npass=True, is_sn=False, is_map=False, is_stds=False, 
                 model_type = 'transencoder2s', device=0):
        super(ModelTransEnc, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = model_type
        self.device = device

        self.seq_len = seq_len
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.d_model = d_model

        self.is_stds = is_stds
        self.is_npass = is_npass
        self.is_sn = is_sn
        self.is_map = is_map
        self.feas_ccs = 2
        if self.is_stds:
            self.feas_ccs += 2
        if self.is_npass:
            self.feas_ccs += 1
        if self.is_sn:
            self.feas_ccs += 1
        if self.is_map:
            self.feas_ccs += 1

        self.seq_embed = nn.Embedding(nvocab, nembed)
        self.src_embed = nn.Sequential(nn.Conv1d(in_channels=nembed+self.feas_ccs, 
                                                 out_channels=self.d_model // 2,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1,
                                                 bias=False),
                                       nn.BatchNorm1d(num_features=self.d_model // 2),
                                       nn.ReLU(inplace=True),
                                       nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
                                       nn.Conv1d(in_channels=self.d_model // 2,
                                                 out_channels=self.d_model,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1,
                                                 bias=False),
                                       nn.BatchNorm1d(num_features=self.d_model),
                                       nn.ReLU(inplace=True),
                                       nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
                                       nn.Conv1d(in_channels=self.d_model,
                                                 out_channels=self.d_model,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1,
                                                 bias=False),
                                       nn.BatchNorm1d(num_features=self.d_model),
                                       nn.ReLU(inplace=True),
                                       nn.MaxPool1d(kernel_size=3, stride=1, padding=1))
        self.src_mask = None

        self.pos_encoder = PositionalEncoding(self.d_model, dropout_rate)
        encoder_layer = TransformerEncoderLayer(d_model=self.d_model, nhead=nhead, dim_feedforward=dim_ff, 
                                                dropout=dropout_rate, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.Linear(self.seq_len * self.d_model, self.d_model)  # TODO: is it suitable?

        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(self.d_model * 2, self.num_classes)

        self.softmax = nn.Softmax(1)

    # def _generate_square_subsequent_mask(self, sz):
    #     mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    #     mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    #     return mask

    def forward(self, kmer, kpass, ipd_means, ipd_stds, pw_means, pw_stds, sns, maps,
                kmer2, kpass2, ipd_means2, ipd_stds2, pw_means2, pw_stds2, sns2, maps2, 
                has_mask=False):
        kmer_embed = self.seq_embed(kmer.long())
        ipd_means = torch.reshape(ipd_means, (-1, self.seq_len, 1)).float()
        pw_means = torch.reshape(pw_means, (-1, self.seq_len, 1)).float()        
        kmer_embed2 = self.seq_embed(kmer2.long())
        ipd_means2 = torch.reshape(ipd_means2, (-1, self.seq_len, 1)).float()
        pw_means2 = torch.reshape(pw_means2, (-1, self.seq_len, 1)).float()

        out1 = torch.cat((kmer_embed, ipd_means, pw_means), 2)  # (N, L, C)
        out2 = torch.cat((kmer_embed2, ipd_means2, pw_means2), 2)  # (N, L, C)

        if self.is_npass:
            kpass = torch.reshape(kpass, (-1, self.seq_len, 1)).float()
            out1 = torch.cat((out1, kpass), 2)  # (N, L, C)
            kpass2 = torch.reshape(kpass2, (-1, self.seq_len, 1)).float()
            out2 = torch.cat((out2, kpass2), 2)  # (N, L, C)
        if self.is_stds:
            ipd_stds = torch.reshape(ipd_stds, (-1, self.seq_len, 1)).float()
            pw_stds = torch.reshape(pw_stds, (-1, self.seq_len, 1)).float()
            out1 = torch.cat((out1, ipd_stds, pw_stds), 2)  # (N, L, C)
            ipd_stds2 = torch.reshape(ipd_stds2, (-1, self.seq_len, 1)).float()
            pw_stds2 = torch.reshape(pw_stds2, (-1, self.seq_len, 1)).float()
            out2 = torch.cat((out2, ipd_stds2, pw_stds2), 2)  # (N, L, C)
        if self.is_sn:
            sns = torch.reshape(sns, (-1, self.seq_len, 1)).float()
            out1 = torch.cat((out1, sns), 2)  # (N, L, C)
            sns2 = torch.reshape(sns2, (-1, self.seq_len, 1)).float()
            out2 = torch.cat((out2, sns2), 2)  # (N, L, C)
        if self.is_map:
            maps = torch.reshape(maps, (-1, self.seq_len, 1)).float()
            out1 = torch.cat((out1, maps), 2)  # (N, L, C)
            maps2 = torch.reshape(maps2, (-1, self.seq_len, 1)).float()
            out2 = torch.cat((out2, maps2), 2)  # (N, L, C)

        out1 = out1.transpose(-1, -2)  # (N, C, L)
        out1 = self.src_embed(out1)  # (N, C, L)
        out1 = out1.transpose(-1, -2)  # (N, L, C)
        out1 = self.pos_encoder(out1)  # (N, L, C) if batch_first else (L, N, C)
        # if has_mask:
        #     device = out1.device
        #     if self.src_mask is None or self.src_mask.size(0) != len(out1):
        #         mask = self._generate_square_subsequent_mask(len(out1)).to(device)
        #         self.src_mask = mask
        # else:
        #     self.src_mask = None
        out1 = self.transformer_encoder(out1, self.src_mask)  # (N, L, C) if batch_first else (L, N, C)
        out1 = out1.reshape(out1.size(0), -1)  # (N, L*C)  # TODO: is it suitable?  
        out1 = self.decoder(out1)  # (N, C)  # TODO: is it suitable?

        out2 = out2.transpose(-1, -2)  # (N, C, L)
        out2 = self.src_embed(out2)  # (N, C, L)
        out2 = out2.transpose(-1, -2)  # (N, L, C)
        out2 = self.pos_encoder(out2)  # (N, L, C) if batch_first else (L, N, C)
        # if has_mask:
        #     device = out2.device
        #     if self.src_mask is None or self.src_mask.size(0) != len(out2):
        #         mask = self._generate_square_subsequent_mask(len(out2)).to(device)
        #         self.src_mask = mask
        # else:
        #     self.src_mask = None
        out2 = self.transformer_encoder(out2, self.src_mask)  # (N, L, C) if batch_first else (L, N, C)
        out2 = out2.reshape(out2.size(0), -1)
        out2 = self.decoder(out2)

        out = torch.cat((out1, out2), 1)
        # output logits
        out = self.relu1(out)  # TODO: is it suitable?
        out = self.dropout1(out)
        out = self.fc1(out)
        return out, self.softmax(out)


class ModelTransEnc2(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""
    def __init__(self, seq_len=21, num_layers=6, num_classes=2,
                 dropout_rate=0.5, d_model=256, nhead=4, dim_ff=512,
                 nvocab=16, nembed=4, 
                 is_npass=True, is_sn=False, is_map=False, is_stds=False, 
                 model_type = 'transencoder2s2', device=0):
        super(ModelTransEnc2, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = model_type
        self.device = device

        self.seq_len = seq_len
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.d_model = d_model

        self.is_stds = is_stds
        self.is_npass = is_npass
        self.is_sn = is_sn
        self.is_map = is_map

        self.feas_ccs = 2
        self.nembed_all = nembed + 2 * NEMBED_KINETICS
        self.seq_embed = nn.Embedding(nvocab, nembed)
        self.ipd_embed = nn.Embedding(MAX_KINETICS + 1, NEMBED_KINETICS)
        self.pw_embed = nn.Embedding(MAX_KINETICS + 1, NEMBED_KINETICS)
        if self.is_stds:
            self.feas_ccs += 2
            self.nembed_all += 2 * NEMBED_KINETICS_STD
            self.ipd_std_embed = nn.Embedding(MAX_KINETICS_STD + 1, NEMBED_KINETICS_STD)
            self.pw_std_embed = nn.Embedding(MAX_KINETICS_STD + 1, NEMBED_KINETICS_STD)
        if self.is_npass:
            self.feas_ccs += 1
            self.nembed_all += NEMBED_PASSES
            self.npass_embed = nn.Embedding(MAX_PASSES, NEMBED_PASSES)
        if self.is_sn:
            self.feas_ccs += 1
            self.nembed_all += NEMBED_SN
            self.sn_embed = nn.Embedding(MAX_SN, NEMBED_SN)
        if self.is_map:
            self.feas_ccs += 1
            self.nembed_all += NEMBED_MAP
            self.map_embed = nn.Embedding(MAX_MAP, NEMBED_MAP)
        self.trans_input = nn.Linear(self.nembed_all, self.d_model, bias=False)
        self.src_mask = None

        self.pos_encoder = PositionalEncoding(self.d_model, dropout_rate)
        encoder_layer = TransformerEncoderLayer(d_model=self.d_model, nhead=nhead, dim_feedforward=dim_ff, 
                                                dropout=dropout_rate, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.Linear(self.seq_len * self.d_model, self.d_model)  # TODO: is it suitable?

        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(self.d_model * 2, self.num_classes)

        self.softmax = nn.Softmax(1)

    # def _generate_square_subsequent_mask(self, sz):
    #     mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    #     mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    #     return mask

    def forward(self, kmer, kpass, ipd_means, ipd_stds, pw_means, pw_stds, sns, maps,
                kmer2, kpass2, ipd_means2, ipd_stds2, pw_means2, pw_stds2, sns2, maps2, 
                has_mask=False):
        kmer_embed = self.seq_embed(kmer.long())
        ipd_means = self.ipd_embed(ipd_means)
        pw_means = self.pw_embed(pw_means)       
        kmer_embed2 = self.seq_embed(kmer2.long())
        ipd_means2 = self.ipd_embed(ipd_means2)
        pw_means2 = self.pw_embed(pw_means2)

        out1 = torch.cat((kmer_embed, ipd_means, pw_means), 2)  # (N, L, C)
        out2 = torch.cat((kmer_embed2, ipd_means2, pw_means2), 2)  # (N, L, C)

        if self.is_npass:
            kpass = self.npass_embed(torch.clamp(kpass, 1, MAX_PASSES))
            out1 = torch.cat((out1, kpass), 2)  # (N, L, C)
            kpass2 = self.npass_embed(torch.clamp(kpass2, 1, MAX_PASSES))
            out2 = torch.cat((out2, kpass2), 2)  # (N, L, C)
        if self.is_stds:
            ipd_stds = self.ipd_std_embed(ipd_stds)
            pw_stds = self.pw_std_embed(pw_stds)
            out1 = torch.cat((out1, ipd_stds, pw_stds), 2)  # (N, L, C)
            ipd_stds2 = self.ipd_std_embed(ipd_stds2)
            pw_stds2 = torch.pw_std_embed(pw_stds2)
            out2 = torch.cat((out2, ipd_stds2, pw_stds2), 2)  # (N, L, C)
        if self.is_sn:
            sns = self.sn_embed(sns)
            out1 = torch.cat((out1, sns), 2)  # (N, L, C)
            sns2 = self.sn_embed(sns2)
            out2 = torch.cat((out2, sns2), 2)  # (N, L, C)
        if self.is_map:
            maps = self.map_embed(maps)
            out1 = torch.cat((out1, maps), 2)  # (N, L, C)
            maps2 = self.map_embed(maps2)
            out2 = torch.cat((out2, maps2), 2)  # (N, L, C)
        out1 = self.trans_input(out1)
        out2 = self.trans_input(out2)

        out1 = self.pos_encoder(out1)  # (N, L, C) if batch_first else (L, N, C)
        # if has_mask:
        #     device = out1.device
        #     if self.src_mask is None or self.src_mask.size(0) != len(out1):
        #         mask = self._generate_square_subsequent_mask(len(out1)).to(device)
        #         self.src_mask = mask
        # else:
        #     self.src_mask = None
        out1 = self.transformer_encoder(out1, self.src_mask)  # (N, L, C) if batch_first else (L, N, C)
        out1 = out1.reshape(out1.size(0), -1)  # (N, L*C)  # TODO: is it suitable?  
        out1 = self.decoder(out1)  # (N, C)  # TODO: is it suitable?

        out2 = out2.transpose(-1, -2)  # (N, C, L)
        out2 = self.src_embed(out2)  # (N, C, L)
        out2 = out2.transpose(-1, -2)  # (N, L, C)
        out2 = self.pos_encoder(out2)  # (N, L, C) if batch_first else (L, N, C)
        # if has_mask:
        #     device = out2.device
        #     if self.src_mask is None or self.src_mask.size(0) != len(out2):
        #         mask = self._generate_square_subsequent_mask(len(out2)).to(device)
        #         self.src_mask = mask
        # else:
        #     self.src_mask = None
        out2 = self.transformer_encoder(out2, self.src_mask)  # (N, L, C) if batch_first else (L, N, C)
        out2 = out2.reshape(out2.size(0), -1)
        out2 = self.decoder(out2)

        out = torch.cat((out1, out2), 1)
        # output logits
        out = self.relu1(out)  # TODO: is it suitable?
        out = self.dropout1(out)
        out = self.fc1(out)
        return out, self.softmax(out)


# Using BS-seq modfreqs as gold standard to train a AggrAttRNN regression model, no softmax
class AggrAttRNN(nn.Module):
    def __init__(self, seq_len=11, num_layers=1, num_classes=1,
                 dropout_rate=0.5, hidden_size=32, binsize=20,
                 model_type="attbigru",
                 device=0):
        super(AggrAttRNN, self).__init__()
        self.model_type = model_type
        self.device = device

        self.seq_len = seq_len
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        self.feas_ccs = binsize + 1
        if self.model_type == "attbilstm":
            self.rnn_cell = "lstm"
            self.rnn = nn.LSTM(self.feas_ccs, self.hidden_size, self.num_layers,
                               dropout=0, batch_first=True, bidirectional=True)
        elif self.model_type == "attbigru":
            self.rnn_cell = "gru"
            self.rnn = nn.GRU(self.feas_ccs, self.hidden_size, self.num_layers,
                              dropout=0, batch_first=True, bidirectional=True)
        else:
            raise ValueError("--model_type not set right!")

        self._att3 = Attention(self.hidden_size * 2, self.hidden_size * 2, self.hidden_size)

        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(self.hidden_size * 2, self.num_classes)  # 2 for bidirection

        # self.softmax = nn.Softmax(1)

    def get_model_type(self):
        return self.model_type

    def init_hidden(self, batch_size, num_layers, hidden_size):
        # Set initial states
        h0 = torch.randn(num_layers * 2, batch_size, hidden_size, requires_grad=True)
        if use_cuda and self.device != "cpu":
            h0 = h0.cuda(self.device)
        if self.rnn_cell == "lstm":
            c0 = torch.randn(num_layers * 2, batch_size, hidden_size, requires_grad=True)
            if use_cuda and self.device != "cpu":
                c0 = c0.cuda(self.device)
            return h0, c0
        return h0

    def forward(self, offsets, histos):

        offsets = torch.reshape(offsets, (-1, self.seq_len, 1)).float()  # (N, L, 1)

        out = torch.cat((histos.float(), offsets), 2)

        out, n_states = self.rnn(out, self.init_hidden(out.size(0),
                                                       self.num_layers,
                                                       self.hidden_size))  # (N, L, nhid*2)
        # attention_net3 ======
        # h_n: (num_layer * 2, N, nhid), h_0, c_0 -> h_n, c_n not affected by batch_first
        # h_n (last layer) = out[:, -1, :self.hidden_size] concats out1[:, 0, self.hidden_size:]
        h_n = n_states[0] if self.rnn_cell == "lstm" else n_states
        h_n = h_n.reshape(self.num_layers, 2, -1, self.hidden_size)[-1]  # last layer (2, N, nhid)
        h_n = h_n.transpose(0, 1).reshape(-1, 1, 2 * self.hidden_size)
        out, att_weights = self._att3(h_n, out)

        out = self.dropout1(out)
        out = self.fc1(out)
        # out = self.softmax(out)

        return out
