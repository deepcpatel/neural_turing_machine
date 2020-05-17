# NTM memory module
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

class memory_unit(nn.Module):

    def __init__(self, N, M, memory_init=None):

        super(memory_unit, self).__init__()

        self.N = N      # Number of Memory cells
        self.M = M      # Single Memory cell length

        self.register_buffer('memory_init', torch.Tensor(N, M))

        # Memory Initialization
        # Following snippet allows user to exclusively give initialization values to the memory
        # Otherwise it initializes automatically
        
        if memory_init == None:
            nn.init.constant_(self.memory_init, 1e-6)   # Memory size is (N, M)
        else:
            self.memory_init = memory_init.clone()

    def memory_size(self):
        return self.N, self.M

    def reset_memory(self, batch_size):
        self.batch_size = batch_size
        self.memory = self.memory_init.clone().repeat(batch_size, 1, 1)

    def memory_read(self, W):                                               # Assuming shape of W is (batch_size x N), Memory -> (batch_size, N, M)
        return torch.bmm(W.unsqueeze(1), self.memory).squeeze(1)            # Out : (batch_size x M) size vector

    def memory_write(self, W, e, a):                                        # Assuming shape of W is (batch_size x N) and shape of e (erase vector) and a (add vector) is (batch_size x M)
        previous_mem = self.memory
        self.memory = torch.Tensor(self.batch_size, self.N, self.M)

        erase_mat = torch.bmm(W.unsqueeze(-1), e.unsqueeze(1))     # Out : (batch_size x N x M) matrix
        add_mat = torch.bmm(W.unsqueeze(-1), a.unsqueeze(1))       # Out : (batch_size x N x M) matrix

        self.memory = previous_mem * (1 - erase_mat) + add_mat     # Out : (batch x N x M) matrix

    def access_memory(self, k, beta, g, s, gamma, W_old):   # Returns the weight vector to access memory

        """
        Input   : 

        k       : Key vector for matching       -> (batch_size x M)
        beta    : Constant for strength focus   -> (batch_size x 1)
        g       : Interpolation gate value      -> (batch_size x 1)
        s       : Shift weightings              -> (batch_size x Some_Length)
        gamma   : Weight sharpening scalar      -> (batch_size x 1)
        W_old   : Previous weight vector        -> (batch_size x N) 
        """

        # Content based addressing
        W_c = self._content_focusing(k, beta)   # Out : (batch_size x N) vector

        # Location based addressing
        W_g = self._gating(W_c, W_old, g)
        W_t = self._shift(W_g, s)
        W = self._sharpen(W_t, gamma)
        return W                                # Out : (batch_size x N) vector

    def _content_focusing(self, key, beta):
        similarity_vector = F.cosine_similarity(key.unsqueeze(1) + 1e-16, self.memory + 1e-16, dim = 2) # We are adding some offset to inputs instead of giving to eps to avoid zero output 
        temp_vec = beta*similarity_vector
        return F.softmax(temp_vec, dim = 1)

    def _gating(self, W_c, W_old, g):
        return g*W_c + (1 - g)*W_old

    def _shift(self, weights, s):  # We assume that len(s) [length of shift vector] is always smaller than or equal to that of len(weights).

        shifted_weights = torch.zeros(weights.shape)
        
        for i in range(self.batch_size):
            shifted_weights[i,:] = self._circ_convolution(weights[i,:], s[i,:])
        return shifted_weights

    def _roll(self, tensor, shift, axis):
        if shift == 0:
            return tensor

        if axis < 0:
            axis += tensor.dim()

        dim_size = tensor.size(axis)
        after_start = dim_size - shift
        if shift < 0:
            after_start = -shift
            shift = dim_size - abs(shift)

        before = tensor.narrow(axis, 0, dim_size - shift)
        after = tensor.narrow(axis, after_start, shift)
        return torch.cat([after, before], axis)

    def _circ_convolution(self, vec1, vec2, start=-1):   # Input : vec1 -> (1 x P) and vec2 -> (1 x Q) and P, Q be any scalars

        # Note : Using Circular Convolution Method taught in DSP Lab 4 (Slides in "Archive" folder)
        #        Circular convolution function is generalized and don't assume anything
        #        Here convolution vector is [-1, 0, 1], thus our start position is -1. It can be changed.

        vec1 = vec1.view(-1, 1)
        vec2 = vec2.view(-1, 1)

        len_1 = vec1.shape[0]
        len_2 = vec2.shape[0]

        if len_1 == len_2:
            c = torch.zeros((len_1, len_1))
        elif len_1 > len_2:
            c = torch.zeros((len_1, len_1))

            temp = torch.zeros(vec1.shape)
            temp[0:len_2, :] = vec2
            vec2 = temp.clone()
        else:
            c = torch.zeros((len_2, len_2))
            
            temp = torch.zeros(vec2.shape)
            temp[0:len_1, :] = vec1
            vec1 = temp.clone()

        vec1 = self._roll(vec1, start, axis = 0)    # Rolling the Original vector based on the start position

        for i in range(vec1.shape[0]):
            c[:, i] = vec1[:,0]
            vec1 = self._roll(vec1, 1, axis = 0)
        return (torch.matmul(c, vec2)).t()   # Out : (1 x P)

    def _sharpen(self, weights, gamma):
        temp_vec = torch.pow(weights, gamma)
        sum_vec = torch.sum(temp_vec, dim=1).view(-1, 1) + 1e-16    # Adding 1e-16 for numerical stability in case of some element being zero
        return torch.div(temp_vec, sum_vec)                         # Out : (batch_size x N) vector