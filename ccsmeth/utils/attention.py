# references:
# https://github.com/Tsedao/MultiRM/blob/master/Scripts/util_layers.py

# https://bastings.github.io/annotated_encoder_decoder/

# https://github.com/mlcommons/training/blob/master/rnn_translator/pytorch/seq2seq/models/attention.py

# https://towardsdatascience.com/attention-seq2seq-with-pytorch-learning-to-invert-a-sequence-34faf4133e53
# https://github.com/b-etienne/Seq2seq-PyTorch
# https://gist.github.com/b-etienne/64f75b980126180dd5e2cd2e92be53fd
import torch
import torch.nn as nn


def mask_3d(inputs, seq_len, mask_value=0.):
    batches = inputs.size()[0]
    assert batches == len(seq_len)
    max_idx = max(seq_len)
    for n, idx in enumerate(seq_len):
        if idx < max_idx.item():
            if len(inputs.size()) == 3:
                inputs[n, idx.int():, :] = mask_value
            else:
                assert len(inputs.size()) == 2, "The size of inputs must be 2 or 3, received {}".format(inputs.size())
                inputs[n, idx.int():] = mask_value
    return inputs


# bahdanau attention
class Attention(nn.Module):
    """
    Inputs:
        last_hidden: (batch_size, hidden_size)  # query, (2, N, C) -> (N, 1, C*2)
        encoder_outputs: (batch_size, max_time, hidden_size)  # key, (N, L, C*2)
    Returns:
        context_vector: (N, 2*C)
        attention_weights: (batch_size, max_time)
    """
    def __init__(self, query_size, key_size, hidden_size=128):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size

        self.Wa = nn.Linear(query_size, hidden_size, bias=False)
        self.Ua = nn.Linear(key_size, hidden_size, bias=False)
        self.va = nn.Linear(hidden_size, 1, bias=False)
        self.attw_softmax = nn.Softmax(1)

    def forward(self, last_hidden, encoder_outputs):

        attention_energies = self._score(last_hidden, encoder_outputs).squeeze(2)  # (N, L, 1) -> (N, L)

        # if seq_len is not None:
        #     attention_energies = mask_3d(attention_energies, seq_len, -float('inf'))

        attention_weights = self.attw_softmax(attention_energies).unsqueeze(2)  # (N, L) -> (N, L, 1)

        values = torch.transpose(encoder_outputs, 1, 2)  # (N, 2*C, L)
        context_vector = torch.matmul(values, attention_weights).squeeze(2)  # (N, 2*C, 1) -> (N, 2*C)

        return context_vector, attention_weights.squeeze(2)

    def _score(self, last_hidden, encoder_outputs):
        """
        Computes an attention score
        :param last_hidden: (batch_size, hidden_dim)  # (2, N, C) -> (N, 1, C*2)
        :param encoder_outputs: (batch_size, max_time, hidden_dim)  # (N, L, C*2)
        :return: a score (batch_size, max_time)
        """
        out = torch.tanh(self.Wa(last_hidden) + self.Ua(encoder_outputs))  # (N, L, nhid)
        return self.va(out)  # (N, L, 1)
