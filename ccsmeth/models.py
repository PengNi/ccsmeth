#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

from abc import ABC

import torch
import torch.nn as nn

from .utils.constants_torch import use_cuda
from .utils.attention import Attention


class ModelAttRNN(nn.Module):
    def __init__(self, seq_len=21, num_layers=3, num_classes=2,
                 dropout_rate=0.5, hidden_size=256,
                 vocab_size=16, embedding_size=4,
                 is_qual=True, is_map=False, is_stds=False, is_npass=False,
                 model_type="attbilstm2s"):
        super(ModelAttRNN, self).__init__()
        self.model_type = model_type

        self.seq_len = seq_len
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        self.embed = nn.Embedding(vocab_size, embedding_size)  # for dna/rna base

        self.is_stds = is_stds
        self.is_npass = is_npass
        self.is_qual = is_qual
        self.is_map = is_map
        self.feas_ccs = 2
        if self.is_stds:
            self.feas_ccs += 2
        if self.is_npass:
            self.feas_ccs += 1
        if self.is_qual:
            self.feas_ccs += 1
        if self.is_map:
            self.feas_ccs += 1
        if self.model_type == "attbilstm2s":
            self.rnn_cell = "lstm"
            self.rnn = nn.LSTM(embedding_size + self.feas_ccs, self.hidden_size, self.num_layers,
                               dropout=dropout_rate, batch_first=True, bidirectional=True)
            self.rnn2 = nn.LSTM(embedding_size + self.feas_ccs, self.hidden_size, self.num_layers,
                                dropout=dropout_rate, batch_first=True, bidirectional=True)
        elif self.model_type == "attbigru2s":
            self.rnn_cell = "gru"
            self.rnn = nn.GRU(embedding_size + self.feas_ccs, self.hidden_size, self.num_layers,
                              dropout=dropout_rate, batch_first=True, bidirectional=True)
            self.rnn2 = nn.GRU(embedding_size + self.feas_ccs, self.hidden_size, self.num_layers,
                               dropout=dropout_rate, batch_first=True, bidirectional=True)
        else:
            raise ValueError("--model_type not set right!")

        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(self.hidden_size * 2 * 2, self.num_classes)  # 2 for bidirection, another 2 for 2 strands

        self._att3 = Attention(self.hidden_size * 2, self.hidden_size * 2, self.hidden_size)
        self._att3_2 = Attention(self.hidden_size * 2, self.hidden_size * 2, self.hidden_size)

        self.softmax = nn.Softmax(1)

    def get_model_type(self):
        return self.model_type

    def init_hidden(self, batch_size, num_layers, hidden_size):
        # Set initial states
        h0 = torch.randn(num_layers * 2, batch_size, hidden_size, requires_grad=True)
        if use_cuda:
            h0 = h0.cuda()
        if self.rnn_cell == "lstm":
            c0 = torch.randn(num_layers * 2, batch_size, hidden_size, requires_grad=True)
            if use_cuda:
                c0 = c0.cuda()
            return h0, c0
        return h0

    def forward(self, kmer, kpass, ipd_means, ipd_stds, pw_means, pw_stds, quals, maps,
                kmer2, kpass2, ipd_means2, ipd_stds2, pw_means2, pw_stds2, quals2, maps2):
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
        if self.is_qual:
            quals = torch.reshape(quals, (-1, self.seq_len, 1)).float()
            out1 = torch.cat((out1, quals), 2)  # (N, L, C)
            quals2 = torch.reshape(quals2, (-1, self.seq_len, 1)).float()
            out2 = torch.cat((out2, quals2), 2)  # (N, L, C)
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
        out2, att_weights2 = self._att3_2(h_n2, out2)

        out = torch.cat((out1, out2), 1)

        out = self.dropout1(out)
        out = self.fc1(out)

        return out, self.softmax(out)
