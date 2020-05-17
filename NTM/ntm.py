# Final NTM version packaginf all the modules
import torch
from torch import nn
from .memory import memory_unit
from .head import NTM_read_head, NTM_write_head
from .controller import controller
from .processor import processor

class NTM_Module(nn.Module):

    def __init__(self, num_inputs, num_outputs, controller_size, controller_layers, num_heads, N, M):

        # Params:
        # num_inputs : Size of input data
        # num_outputs : Size of output data
        # controller_size : Size of LSTM Controller output/state
        # controller_layers : Number of layers in LSTM Network
        # num_heads : Number of Read and Write heads to be created
        # N : Number of memory cells
        # M : Size of Each memory cell

        super(NTM_Module, self).__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.controller_size = controller_size
        self.controller_layers = controller_layers
        self.num_heads = num_heads
        self.N = N
        self.M = M

        # Creating NTM modules
        self.memory = memory_unit(self.N, self.M)
        self.controller = controller(self.num_inputs + self.M*self.num_heads, self.controller_size, self.controller_layers)
        heads = nn.ModuleList([])
        for i in range(self.num_heads):
            heads += [NTM_read_head(self.memory, self.controller_size), NTM_write_head(self.memory, self.controller_size)]

        self.processor = processor(self.num_outputs, self.M, self.controller, heads)

    def initialization(self, batch_size):   # Initializing all the Modules
        self.batch_size = batch_size
        self.memory.reset_memory(batch_size)
        self.previous_state = self.processor.create_new_state(batch_size)

    def forward(self, X=None):
        if X is None:
            X = torch.zeros(self.batch_size, self.num_inputs)
        out, self.previous_state = self.processor(X, self.previous_state)
        return out, self.previous_state

    def calculate_num_params(self):     # This maybe for model statistics. Adapted
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)
        return num_params