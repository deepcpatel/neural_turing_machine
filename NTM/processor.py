# Processor for Neural Turing Machine
import torch
from torch import nn
import torch.nn.functional as F

class processor(nn.Module):
    def __init__(self, output_size, memory_M, controller, heads):
        # Parameters:
        # output_size -> Controller Output Size
        # memory_M -> Width of each strip of Memory
        # controller -> Controller object
        # heads -> Head object list

        super(processor, self).__init__()

        self.heads = heads
        self.controller = controller

        _, controller_size = self.controller.network_size()
        
        # Initializing read head values
        self.num_read_heads = 0
        self.init_r = []

        for head in heads:
            if head.head_type() == 'R':
                temp = torch.randn(1, memory_M)*0.01
                self.register_buffer("read_bias_" + str(self.num_read_heads), temp.data)
                self.init_r += [temp]
                self.num_read_heads += 1

        assert self.num_read_heads > 0, "Read Heads must be atleast 1"

        self.proc = nn.Linear(controller_size + self.num_read_heads*memory_M, output_size)

        # Initializing the Parameters for Linear Layers
        nn.init.xavier_uniform_(self.proc.weight, gain=1)
        nn.init.normal_(self.proc.bias, std=0.01)

    def create_new_state(self, batch_size):     # Re-creates the New States
        init_r = [r.clone().repeat(batch_size, 1) for r in self.init_r]
        controller_state = self.controller.create_hidden_state(batch_size)
        heads_state = [head.create_new_state(batch_size) for head in self.heads]
        return init_r, controller_state, heads_state

    def forward(self, X, prev_state):   # X dimensions -> (batch_size x num_inputs)

        # Previous State Unpacking:
        prev_read, prev_controller_state, prev_head_weights = prev_state

        # prev_read[i] -> batch_size x M
        # prev_read -> batch_size x (M*no_read_heads) 

        # Getting embeddings from controller
        # Making input for controller
        inp = torch.cat([X] + prev_read, dim = 1)   # inp -> (batch_size x (num_inputs + (no_read_heads*M))
        c_output, c_state = self.controller(inp, prev_controller_state)

        # Reading and Writing in the Memory
        read_vec = []
        head_weights = []

        for head, prev_weights in zip(self.heads, prev_head_weights):
            if head.head_type() == 'R':
                weights, read = head(c_output, prev_weights)
                read_vec += [read]
            else:
                weights = head(c_output, prev_weights)
            head_weights += [weights]

        # Packing State Vectors for Next step
        curr_state = (read_vec, c_state, head_weights)

        # Generating Output
        inp2 = torch.cat([c_output] + read_vec, dim = 1)
        out = self.proc(inp2)
        return out, curr_state
