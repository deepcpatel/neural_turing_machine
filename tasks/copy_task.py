import torch
from torch import nn
from torch import optim

import numpy as np
import random

# from NTM.ntm import NTM_Module            # Original Implementation
from NTM_stable.ntm import NTM_Module       # Stable implementation as given in "Implementing Neural Turing Machines" paper

class task_copy():

    def __init__(self):
        self.name = "copy_task"
        self.controller_size = 100 
        self.controller_layers = 1
        self.num_heads = 1
        self.sequence_width = 8
        self.sequence_min_len = 1
        self.sequence_max_len = 2   # Default: 20
        self.memory_N = 128
        self.memory_M = 20
        self.num_batches = 1000
        self.batch_size = 1
        self.rmsprop_lr = 1e-4
        self.rmsprop_momentum = 0.9
        self.rmsprop_alpha = 0.95
        self.machine = None
        self.loss = None
        self.optimizer = None

    def _data_maker(self, num_batches, batch_size, seq_width, min_len, max_len):    # Generates data for copy task
        # The input data vector will be of size (num_data_rows x batch_size x num_data_columns)
        #
        # 1 1 1 0 1  | 1 1 0 1 0 | 1 1 1 0 1 | 1 0 1 1 1
        # 0 0 1 0 1  | 0 1 0 1 1 | 0 1 0 0 1 | 0 0 1 1 0
        # 0 1 1 0 1  | 1 1 0 0 0 | 1 0 1 0 1 | 0 0 1 1 0
        #
        # Above is the example of data. num_data_rows = 3, num_data_columns = 5, batch_size = 4
        #
        # At a time we will give each row strip to the NTM for prediction as shown below. Therefore input size for one interaction will be (batch_size x num_data_columns)
        # 
        # 1 1 1 0 1  | 1 1 0 1 0 | 1 1 1 0 1 | 1 0 1 1 1
        
        for batch_num in range(num_batches):
            # All batches have the same sequence length
            seq_len = random.randint(min_len, max_len)
            seq = np.random.binomial(1, 0.5, (seq_len, batch_size, seq_width))  # Here, seq_len = num_data_rows and seq_width = num_data_columns
            seq = torch.from_numpy(seq)

            # The input includes an additional channel used for the delimiter
            inp = torch.zeros(seq_len + 1, batch_size, seq_width + 1)
            inp[:seq_len, :, :seq_width] = seq
            inp[seq_len, :, seq_width] = 1.0 # delimiter in our control channel
            outp = seq.clone()

            yield batch_num+1, inp.float(), outp.float()

    def init_ntm(self):
        self.machine = NTM_Module(self.sequence_width + 1, self.sequence_width, self.controller_size, self.controller_layers, self.num_heads, self.memory_N, self.memory_M)

    def init_loss(self):
        self.loss = nn.BCEWithLogitsLoss()

    def init_optimizer(self):
        self.optimizer = optim.RMSprop(self.machine.parameters(), momentum = self.rmsprop_momentum, alpha = self.rmsprop_alpha, lr = self.rmsprop_lr)

    def calc_loss(self, Y_pred, Y):
        return self.loss(Y_pred, Y)

    def get_sample_data(self):  # Sample data for Testing
        batch_size = 1
        seq_len = random.randint(self.sequence_min_len, self.sequence_max_len)
        seq = np.random.binomial(1, 0.5, (seq_len, batch_size, self.sequence_width))  # Here, seq_len = num_data_rows and seq_width = num_data_columns
        seq = torch.from_numpy(seq)

        # The input includes an additional channel used for the delimiter
        inp = torch.zeros(seq_len + 1, batch_size, self.sequence_width + 1)
        inp[:seq_len, :, :self.sequence_width] = seq
        inp[seq_len, :, self.sequence_width] = 1.0 # delimiter in our control channel
        outp = seq.clone()

        return inp.float(), outp.float()

    def get_training_data(self):
        return self._data_maker(self.num_batches, self.batch_size, self.sequence_width, self.sequence_min_len, self.sequence_max_len)