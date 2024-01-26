#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import torch
import torch.nn as nn

from .utils.constants_torch import use_cuda
from .utils.attention import Attention

from .utils.process_utils import N_VOCAB, MAX_KINETICS, MAX_PASSES, MAX_MAP
from .utils.process_utils import NEMBED_BASE, NEMBED_KINETICS, NEMBED_PASSES, NEMBED_MAP, NEMBED_SN, NEMBED_KINETICS_STD

import math

class ModelAttRNN(nn.Module):
    def __init__(self, seq_len=21, num_layers=3, num_classes=2,
                 dropout_rate=0.5, hidden_size=256,
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
        self.n_embed = NEMBED_BASE

        self.embed = nn.Embedding(N_VOCAB, self.n_embed)  # for dna/rna base

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
            self.feas_ccs += 4
        if self.is_map:
            self.feas_ccs += 1
        if self.model_type == "attbilstm2s":
            self.rnn_cell = "lstm"
            self.rnn = nn.LSTM(self.n_embed + self.feas_ccs, self.hidden_size, self.num_layers,
                               dropout=dropout_rate, batch_first=True, bidirectional=True)
        elif self.model_type == "attbigru2s":
            self.rnn_cell = "gru"
            self.rnn = nn.GRU(self.n_embed + self.feas_ccs, self.hidden_size, self.num_layers,
                              dropout=dropout_rate, batch_first=True, bidirectional=True)
        else:
            raise ValueError("--model_type not set right!")

        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(self.hidden_size * 2 * 2, self.num_classes)  # 2 for bidirection, another 2 for 2 strands

        self._att3 = Attention(self.hidden_size * 2, self.hidden_size * 2, self.hidden_size)

        self.softmax = nn.Softmax(1)

        self.init_weights()

    def get_model_type(self):
        return self.model_type
    
    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.embed.weight, -initrange, initrange)
        nn.init.zeros_(self.fc1.bias)
        nn.init.uniform_(self.fc1.weight, -initrange, initrange)

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
        kmer_embed = self.embed(kmer.int())

        ipd_means = torch.reshape(ipd_means, (-1, self.seq_len, 1)).float()
        pw_means = torch.reshape(pw_means, (-1, self.seq_len, 1)).float()
        kmer_embed2 = self.embed(kmer2.int())
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
            sns = sns.unsqueeze(1).expand(-1, self.seq_len, -1).float()
            out1 = torch.cat((out1, sns), 2)  # (N, L, C)
            sns2 = sns2.unsqueeze(1).expand(-1, self.seq_len, -1).float()
            out2 = torch.cat((out2, sns2), 2)  # (N, L, C)
        if self.is_map:
            maps = torch.reshape(maps, (-1, self.seq_len, 1)).float()
            out1 = torch.cat((out1, maps), 2)  # (N, L, C)
            maps2 = torch.reshape(maps2, (-1, self.seq_len, 1)).float()
            out2 = torch.cat((out2, maps2), 2)  # (N, L, C)

        out1, n_states1 = self.rnn(out1, self.init_hidden(out1.size(0),
                                                          self.num_layers,
                                                          self.hidden_size))  # (N, L, nhid*2)
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
        out2, att_weights2 = self._att3(h_n2, out2)

        out = torch.cat((out1, out2), 1)

        out = self.dropout1(out)
        out = self.fc1(out)

        return out, self.softmax(out)


class EmbedBlockPlus(nn.Module):
    def __init__(self, d_model=4, kernel_size=3, stride=1, padding=1, bias=False):
        super(EmbedBlockPlus, self).__init__()
        self.d_model = d_model
        self.conv_embed = nn.Sequential(nn.Conv1d(in_channels=d_model,
                                                  out_channels=d_model,
                                                  kernel_size=kernel_size,
                                                  stride=stride,
                                                  padding=padding,
                                                  bias=bias),
                                        nn.BatchNorm1d(num_features=d_model),
                                        nn.ReLU(inplace=True),
                                        nn.MaxPool1d(kernel_size=kernel_size,
                                                     stride=stride,
                                                     padding=padding)
                                        )
    def forward(self, x):  # input (N, C, L)
        return self.conv_embed(x)


class SrcEmbed(nn.Module):  # for src_embed, no dropout
    def __init__(self, intpu_dim=1, d_model=4, block_plus=1, kernel_size=3, stride=1, padding=1, bias=False):
        super(SrcEmbed, self).__init__()
        self.d_model = d_model
        # self.fc1 = nn.Linear(intpu_dim, d_model)
        self.conv_embed = nn.Sequential(nn.Conv1d(in_channels=intpu_dim, 
                                                  out_channels=self.d_model // 2,
                                                  kernel_size=kernel_size,
                                                  stride=stride,
                                                  padding=padding,
                                                  bias=bias),
                                        nn.BatchNorm1d(num_features=self.d_model // 2),
                                        nn.ReLU(inplace=True),
                                        nn.MaxPool1d(kernel_size=kernel_size,
                                                        stride=stride,
                                                        padding=padding),
                                        nn.Conv1d(in_channels=self.d_model // 2,
                                                    out_channels=self.d_model,
                                                    kernel_size=kernel_size,
                                                    stride=stride,
                                                    padding=padding,
                                                    bias=bias),
                                        nn.BatchNorm1d(num_features=self.d_model),
                                        nn.ReLU(inplace=True),
                                        nn.MaxPool1d(kernel_size=kernel_size,
                                                        stride=stride,
                                                        padding=padding)
                                        )
        self.conv_embed_plus = None
        if block_plus >= 1:
            layers = []
            for i in range(block_plus):
                layers.append(EmbedBlockPlus(d_model=self.d_model,
                                             kernel_size=kernel_size,
                                             stride=stride,
                                             padding=padding,
                                             bias=bias))
            self.conv_embed_plus = nn.Sequential(*layers)
    
    def forward(self, x):  # input (N, L, C)
        x = x.transpose(-1, -2)  # (N, C, L)
        x = self.conv_embed(x)
        if self.conv_embed_plus is not None:
            x = self.conv_embed_plus(x)
        x = x.transpose(-1, -2)  # (N, L, C)
        return x


class ModelAttRNN2(nn.Module):
    def __init__(self, seq_len=21, num_layers=3, num_classes=2,
                 dropout_rate=0.5, hidden_size=256,
                 is_npass=True, is_sn=False, is_map=False, is_stds=False, 
                 model_type="attbigru2s2",
                 device=0):
        super(ModelAttRNN2, self).__init__()
        self.model_type = model_type
        self.device = device

        self.seq_len = seq_len
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        self.is_stds = is_stds
        self.is_npass = is_npass
        self.is_sn = is_sn
        self.is_map = is_map

        self.feas_ccs = 2
        self.nembed_all = NEMBED_BASE + 2 * NEMBED_KINETICS
        self.seq_embed = nn.Embedding(N_VOCAB, NEMBED_BASE)  # for dna/rna base
        self.ipd_embed = nn.Embedding(MAX_KINETICS + 1, NEMBED_KINETICS)
        self.pw_embed = nn.Embedding(MAX_KINETICS + 1, NEMBED_KINETICS)
        if self.is_stds:
            self.feas_ccs += 2
            self.nembed_all += 2 * NEMBED_KINETICS_STD
            self.ipd_std_embed = SrcEmbed(1, NEMBED_KINETICS_STD, block_plus=1, 
                                          kernel_size=3, stride=1, padding=1, bias=False)
            self.pw_std_embed = SrcEmbed(1, NEMBED_KINETICS_STD, block_plus=1,
                                         kernel_size=3, stride=1, padding=1, bias=False)
        if self.is_npass:
            self.feas_ccs += 1
            self.nembed_all += NEMBED_PASSES
            self.npass_embed = nn.Embedding(MAX_PASSES + 1, NEMBED_PASSES)
        if self.is_sn:
            self.feas_ccs += 1
            self.nembed_all += NEMBED_SN
            self.sn_embed = SrcEmbed(4, NEMBED_SN, block_plus=0,
                                     kernel_size=3, stride=1, padding=1, bias=False)
        if self.is_map:
            self.feas_ccs += 1
            self.nembed_all += NEMBED_MAP
            self.map_embed = nn.Embedding(MAX_MAP, NEMBED_MAP)
        
        if self.model_type == "attbilstm2s2":
            self.rnn_cell = "lstm"
            self.rnn = nn.LSTM(self.nembed_all, self.hidden_size, self.num_layers,
                               dropout=dropout_rate, batch_first=True, bidirectional=True)
        elif self.model_type == "attbigru2s2":
            self.rnn_cell = "gru"
            self.rnn = nn.GRU(self.nembed_all, self.hidden_size, self.num_layers,
                              dropout=dropout_rate, batch_first=True, bidirectional=True)
        else:
            raise ValueError("--model_type not set right!")

        self._att3 = Attention(self.hidden_size * 2, self.hidden_size * 2, self.hidden_size)

        self.classifier = nn.Sequential(nn.Linear(self.hidden_size * 2 * 2, self.hidden_size * 2 * 2),
                                        nn.ReLU(),
                                        nn.Dropout(p=dropout_rate),
                                        nn.Linear(self.hidden_size * 2 * 2, self.num_classes))

        self.softmax = nn.Softmax(1)

        self.init_weights()

    def get_model_type(self):
        return self.model_type
    
    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.seq_embed.weight, -initrange, initrange)
        nn.init.uniform_(self.ipd_embed.weight, -initrange, initrange)
        nn.init.uniform_(self.pw_embed.weight, -initrange, initrange)
        if self.is_stds:
            None  # placeholder
        if self.is_npass:
            nn.init.uniform_(self.npass_embed.weight, -initrange, initrange)
        if self.is_sn:
            None  # placeholder
        if self.is_map:
            nn.init.uniform_(self.map_embed.weight, -initrange, initrange)
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -initrange, initrange)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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
        kmer_embed = self.seq_embed(kmer.int())
        ipd_means = self.ipd_embed(ipd_means.int())
        pw_means = self.pw_embed(pw_means.int())
        kmer_embed2 = self.seq_embed(kmer2.int())
        ipd_means2 = self.ipd_embed(ipd_means2.int())
        pw_means2 = self.pw_embed(pw_means2.int())

        out1 = torch.cat((kmer_embed, ipd_means, pw_means), 2)  # (N, L, C)
        out2 = torch.cat((kmer_embed2, ipd_means2, pw_means2), 2)  # (N, L, C)

        if self.is_npass:
            kpass = self.npass_embed(torch.clamp(kpass, 1, MAX_PASSES).int())
            out1 = torch.cat((out1, kpass), 2)  # (N, L, C)
            kpass2 = self.npass_embed(torch.clamp(kpass2, 1, MAX_PASSES).int())
            out2 = torch.cat((out2, kpass2), 2)  # (N, L, C)
        if self.is_stds:
            ipd_stds = self.ipd_std_embed(torch.reshape(ipd_stds, (-1, self.seq_len, 1)).float())
            pw_stds = self.pw_std_embed(torch.reshape(pw_stds, (-1, self.seq_len, 1)).float())
            out1 = torch.cat((out1, ipd_stds, pw_stds), 2)  # (N, L, C)
            ipd_stds2 = self.ipd_std_embed(torch.reshape(ipd_stds2, (-1, self.seq_len, 1)).float())
            pw_stds2 = self.pw_std_embed(torch.reshape(pw_stds2, (-1, self.seq_len, 1)).float())
            out2 = torch.cat((out2, ipd_stds2, pw_stds2), 2)  # (N, L, C)
        if self.is_sn:
            sns = self.sn_embed(sns.unsqueeze(1).expand(-1, self.seq_len, -1).float())
            out1 = torch.cat((out1, sns), 2)  # (N, L, C)
            sns2 = self.sn_embed(sns2.unsqueeze(1).expand(-1, self.seq_len, -1).float())
            out2 = torch.cat((out2, sns2), 2)  # (N, L, C)
        if self.is_map:
            maps = self.map_embed(maps)
            out1 = torch.cat((out1, maps), 2)  # (N, L, C)
            maps2 = self.map_embed(maps2)
            out2 = torch.cat((out2, maps2), 2)  # (N, L, C)

        out1, n_states1 = self.rnn(out1, self.init_hidden(out1.size(0),
                                                          self.num_layers,
                                                          self.hidden_size))  # (N, L, nhid*2)
        out2, n_states2 = self.rnn(out2, self.init_hidden(out2.size(0),
                                                          self.num_layers,
                                                          self.hidden_size))  # (N, L, nhid*2)

        # attention_net3 ======
        # h_n: (num_layer * 2, N, nhid), h_0, c_0 -> h_n, c_n not affected by batch_first
        # h_n (last layer) = out[:, -1, :self.hidden_size] concats out1[:, 0, self.hidden_size:]
        h_n1 = n_states1[0] if self.rnn_cell == "lstm" else n_states1
        h_n1 = h_n1.reshape(self.num_layers, 2, -1, self.hidden_size)[-1]  # last layer (2, N, nhid)
        h_n1 = h_n1.transpose(0, 1).reshape(-1, 1, 2 * self.hidden_size)
        out1, _ = self._att3(h_n1, out1)

        h_n2 = n_states2[0] if self.rnn_cell == "lstm" else n_states2
        h_n2 = h_n2.reshape(self.num_layers, 2, -1, self.hidden_size)[-1]  # last layer (2, N, nhid)
        h_n2 = h_n2.transpose(0, 1).reshape(-1, 1, 2 * self.hidden_size)
        out2, _ = self._att3(h_n2, out2)

        out = torch.cat((out1, out2), 1)

        out = self.classifier(out)

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


class PositionalEmbedding(nn.Module):
    def __init__(self, seq_len, d_model, dropout=0.1):
        super(PositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.seq_len = seq_len
        self.pos_embed = nn.Embedding(seq_len, d_model)
    
    def forward(self, x):
        pos = torch.arange(0, self.seq_len, dtype=torch.int, device=x.device)
        pos = pos.unsqueeze(0).expand(x.size(0), -1)
        x = x + self.pos_embed(pos)
        return self.dropout(x)


class ModelTransEnc(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""
    def __init__(self, seq_len=21, num_layers=6, num_classes=2,
                 dropout_rate=0.5, d_model=256, nhead=4, dim_ff=512,
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
        self.nembed_all = NEMBED_BASE + 2 * NEMBED_KINETICS
        self.seq_embed = nn.Embedding(N_VOCAB, NEMBED_BASE)
        self.ipd_embed = nn.Embedding(MAX_KINETICS + 1, NEMBED_KINETICS)
        self.pw_embed = nn.Embedding(MAX_KINETICS + 1, NEMBED_KINETICS)
        if self.is_stds:
            self.feas_ccs += 2
            self.nembed_all += 2 * NEMBED_KINETICS_STD
            self.ipd_std_embed = SrcEmbed(1, NEMBED_KINETICS_STD, block_plus=1, 
                                          kernel_size=3, stride=1, padding=1, bias=False)
            self.pw_std_embed = SrcEmbed(1, NEMBED_KINETICS_STD, block_plus=1,
                                         kernel_size=3, stride=1, padding=1, bias=False)
        if self.is_npass:
            self.feas_ccs += 1
            self.nembed_all += NEMBED_PASSES
            self.npass_embed = nn.Embedding(MAX_PASSES + 1, NEMBED_PASSES)
        if self.is_sn:
            self.feas_ccs += 1
            self.nembed_all += NEMBED_SN
            self.sn_embed = SrcEmbed(4, NEMBED_SN, block_plus=0,
                                     kernel_size=3, stride=1, padding=1, bias=False)
        if self.is_map:
            self.feas_ccs += 1
            self.nembed_all += NEMBED_MAP
            self.map_embed = nn.Embedding(MAX_MAP, NEMBED_MAP)
        # self.trans_input = nn.Linear(self.nembed_all, self.d_model, bias=False)
        self.trans_input = SrcEmbed(self.nembed_all, self.d_model, block_plus=1, 
                                    kernel_size=3, stride=1, padding=1, bias=False)
        self.src_mask = None

        # self.pos_encoder = PositionalEncoding(self.d_model, dropout_rate)
        self.pos_encoder = PositionalEmbedding(self.seq_len, self.d_model, dropout_rate)
        encoder_layer = TransformerEncoderLayer(d_model=self.d_model, nhead=nhead, dim_feedforward=dim_ff, 
                                                dropout=dropout_rate, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)
        # self.decoder = nn.Sequential(nn.Linear(self.seq_len, 1, bias=False),
        #                              nn.Flatten(1, 2))
        # self.decoder = Attention(self.d_model, self.d_model, self.d_model)

        self.classifier = nn.Sequential(nn.Linear(self.d_model * 2, self.d_model * 2),
                                        nn.ReLU(),
                                        nn.Dropout(p=dropout_rate),
                                        nn.Linear(self.d_model * 2, self.num_classes))

        self.softmax = nn.Softmax(1)

        self.init_weights()

    # def _generate_square_subsequent_mask(self, sz):
    #     mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    #     mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    #     return mask
    
    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.seq_embed.weight, -initrange, initrange)
        nn.init.uniform_(self.ipd_embed.weight, -initrange, initrange)
        nn.init.uniform_(self.pw_embed.weight, -initrange, initrange)
        if self.is_stds:
            None  # placeholder
        if self.is_npass:
            nn.init.uniform_(self.npass_embed.weight, -initrange, initrange)
        if self.is_sn:
            None  # placeholder
        if self.is_map:
            nn.init.uniform_(self.map_embed.weight, -initrange, initrange)
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -initrange, initrange)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if isinstance(self.pos_encoder, nn.Linear):
            nn.init.uniform_(self.pos_encoder.weight, -initrange, initrange)
            if self.pos_encoder.bias is not None:
                nn.init.zeros_(self.pos_encoder.bias)

    def forward(self, kmer, kpass, ipd_means, ipd_stds, pw_means, pw_stds, sns, maps,
                kmer2, kpass2, ipd_means2, ipd_stds2, pw_means2, pw_stds2, sns2, maps2, 
                has_mask=False):
        kmer_embed = self.seq_embed(kmer.int())
        ipd_means = self.ipd_embed(ipd_means.int())
        pw_means = self.pw_embed(pw_means.int())
        kmer_embed2 = self.seq_embed(kmer2.int())
        ipd_means2 = self.ipd_embed(ipd_means2.int())
        pw_means2 = self.pw_embed(pw_means2.int())

        out1 = torch.cat((kmer_embed, ipd_means, pw_means), 2)  # (N, L, C)
        out2 = torch.cat((kmer_embed2, ipd_means2, pw_means2), 2)  # (N, L, C)

        if self.is_npass:
            kpass = self.npass_embed(torch.clamp(kpass, 1, MAX_PASSES).int())
            out1 = torch.cat((out1, kpass), 2)  # (N, L, C)
            kpass2 = self.npass_embed(torch.clamp(kpass2, 1, MAX_PASSES).int())
            out2 = torch.cat((out2, kpass2), 2)  # (N, L, C)
        if self.is_stds:
            ipd_stds = self.ipd_std_embed(torch.reshape(ipd_stds, (-1, self.seq_len, 1)).float())
            pw_stds = self.pw_std_embed(torch.reshape(pw_stds, (-1, self.seq_len, 1)).float())
            out1 = torch.cat((out1, ipd_stds, pw_stds), 2)  # (N, L, C)
            ipd_stds2 = self.ipd_std_embed(torch.reshape(ipd_stds2, (-1, self.seq_len, 1)).float())
            pw_stds2 = self.pw_std_embed(torch.reshape(pw_stds2, (-1, self.seq_len, 1)).float())
            out2 = torch.cat((out2, ipd_stds2, pw_stds2), 2)  # (N, L, C)
        if self.is_sn:
            sns = self.sn_embed(sns.unsqueeze(1).expand(-1, self.seq_len, -1).float())
            out1 = torch.cat((out1, sns), 2)  # (N, L, C)
            sns2 = self.sn_embed(sns2.unsqueeze(1).expand(-1, self.seq_len, -1).float())
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
        # out1 = self.decoder(out1.transpose(1, 2))  # (N, C)
        # out1_mean = torch.mean(out1, dim=1)  # (N, C)
        # out1, _ = self.decoder(out1_mean.unsqueeze(1), out1)  # (N, C)
        out1 = torch.mean(out1, dim=1)  # (N, C)

        out2 = self.pos_encoder(out2)  # (N, L, C) if batch_first else (L, N, C)
        # if has_mask:
        #     device = out2.device
        #     if self.src_mask is None or self.src_mask.size(0) != len(out2):
        #         mask = self._generate_square_subsequent_mask(len(out2)).to(device)
        #         self.src_mask = mask
        # else:
        #     self.src_mask = None
        out2 = self.transformer_encoder(out2, self.src_mask)  # (N, L, C) if batch_first else (L, N, C)
        # out2 = self.decoder(out2.transpose(1, 2))
        # out2_mean = torch.mean(out2, dim=1)  # (N, C)
        # out2, _ = self.decoder(out2_mean.unsqueeze(1), out2)  # (N, C)
        out2 = torch.mean(out2, dim=1)  # (N, C)

        out = torch.cat((out1, out2), 1)
        # output logits
        out = self.classifier(out)
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
