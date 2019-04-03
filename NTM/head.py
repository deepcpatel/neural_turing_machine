# NTM Read and Write head module

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# Following split() function is taken from GitHub user loudinthecloud's NTM implementation it splits the head_output according to split criteria (split_plan) to separate out memory parameters
def split(head_out, split_plan):
    assert head_out.size()[1] == sum(split_plan), "Head output length must be equal to that specified before in read_lengths"
    l = np.cumsum([0] + split_plan)
    results = []
    for s, e in zip(l[:-1], l[1:]):
        results += [head_out[:, s:e]]
    return results

class NTM_read_head(nn.Module):     # Read Head for Memory

    def __init__(self, memory, controller_size):

        super(NTM_read_head, self).__init__()

        self.memory = memory
        self.N, self.M = memory.memory_size()
        self.controller_size = controller_size  # controller_size is the size of RNN network embeddings

        # k, beta, g, s, gamma sizes as defined in paper
        self.read_lengths = [self.M, 1, 1, 3, 1]

        # Defining head output as a linear layer, which is intended to be given to memory and accepts LSTM embeddings as input
        self.head_output = nn.Linear(self.controller_size, np.sum(self.read_lengths))

        # Initialize the linear Parameters
        nn.init.xavier_uniform_(self.head_output.weight, gain=1.4)
        nn.init.normal_(self.head_output.bias, std=0.01)

    def head_type(self):
        return "R"      # Head Type : (R)ead

    def create_new_state(self, batch_size): # The State holds the Weights
        return torch.zeros(batch_size, self.N)

    def fine_tune_params(self, k, beta, g, s, gamma):   # Fine tunes the params according to the criteria given in the paper
        k = k.clone()
        beta = F.softplus(beta)
        g = torch.sigmoid(g)
        s = F.softmax(s, dim=1)
        gamma = 1 + F.softplus(gamma)
        return k, beta, g, s, gamma

    def forward(self, rnn_embeddings, W_old):
        out = self.head_output(rnn_embeddings)
        k, beta, g, s, gamma = split(out, self.read_lengths)
        k, beta, g, s, gamma = self.fine_tune_params(k, beta, g, s, gamma)
        W = self.memory.access_memory(k, beta, g, s, gamma, W_old)
        mem_content = self.memory.memory_read(W)
        return W, mem_content

class NTM_write_head(nn.Module):    # Write Head for Memory
    def __init__(self, memory, controller_size):

        super(NTM_write_head, self).__init__()

        self.memory = memory
        self.N, self.M = memory.memory_size()
        self.controller_size = controller_size  # controller_size is the size of RNN network embeddings

        # k, beta, g, s, gamma, e,a sizes as defined in paper
        self.write_lengths = [self.M, 1, 1, 3, 1, self.M, self.M]

        # Defining head output  as a linear layer, which is intended to be given to memory and accepts LSTM embeddings as input
        self.head_output = nn.Linear(self.controller_size, np.sum(self.write_lengths))

        # Initialize the linear Parameters
        nn.init.xavier_uniform_(self.head_output.weight, gain=1.4)
        nn.init.normal_(self.head_output.bias, std=0.01)

    def head_type(self):
        return "W"          # Head Type : (W)rite

    def create_new_state(self, batch_size): # The State holds the Weights
        return torch.zeros(batch_size, self.N)

    def fine_tune_params(self, k, beta, g, s, gamma, e):   # Fine tunes the params according to the criteria given in the paper
        k = k.clone()
        beta = F.softplus(beta)
        g = torch.sigmoid(g)
        s = F.softmax(s, dim=1)
        gamma = 1 + F.softplus(gamma)
        e = torch.sigmoid(e)
        return k, beta, g, s, gamma, e

    def forward(self, rnn_embeddings, W_old):
        out = self.head_output(rnn_embeddings)
        k, beta, g, s, gamma, e, a = split(out, self.write_lengths)
        k, beta, g, s, gamma, e = self.fine_tune_params(k, beta, g, s, gamma, e)
        W = self.memory.access_memory(k, beta, g, s, gamma, W_old)
        self.memory.memory_write(W, e, a)
        return W
