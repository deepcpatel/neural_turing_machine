import pytest
import torch
from NTM.memory import memory_unit

# Testing script by loudinthecloud

def _t(*l):
    return torch.Tensor(l).unsqueeze(0)

class TestMemoryReadWrite:
    N = 4
    M = 4

    def setup_class(self):
        self.memory = memory_unit(self.N, self.M)
        self.memory.reset_memory(batch_size=1)

    def teardown_class(self):
        del self.memory

    def test_size(self):
        n, m = self.memory.memory_size()
        assert n == self.N
        assert m == self.M

    @pytest.mark.parametrize('w, e, a, expected', [
        (_t(1, 0, 0, 0), _t(1, 1, 1, 1), _t(1, 0, 0, 0), _t(1, 0, 0, 0)),
        (_t(0, 1, 0, 0), _t(1, 1, 1, 1), _t(0, 1, 0, 0), _t(0, 1, 0, 0)),
        (_t(0, 0, 1, 0), _t(1, 1, 1, 1), _t(0, 0, 1, 0), _t(0, 0, 1, 0)),
        (_t(0, 0, 0, 1), _t(1, 1, 1, 1), _t(0, 0, 0, 1), _t(0, 0, 0, 1)),
        (_t(1, 0, 0, 0), _t(0, 1, 1, 1), _t(0, 1, 1, 1), _t(1, 1, 1, 1)),
        (_t(0, 1, 0, 0), _t(0, 0, 0, 0), _t(0, 0, 1, 0), _t(0, 1, 1, 0)),
        (_t(0, 0, 1, 0), _t(0, 0, 0, 0), _t(0, 0, 0, 0), _t(0, 0, 1, 0)),
        (_t(0, 0, 0, 1), _t(0, 0, 0, 0.5), _t(0, 0, 0, 0.2), _t(0, 0, 0, 0.7)),
        (_t(0.5, 0.5, 0, 0), _t(1, 1, 1, 1), _t(0, 0, 0, 0), _t(0.25, 0.5, 0.5, 0.25)),
    ])
    def test_read_write(self, w, e, a, expected):
        self.memory.memory_write(w, e, a)
        result = self.memory.memory_read(w)
        assert torch.equal(expected.data, result.data)


@pytest.fixture
def mem():
    mm = memory_unit(4, 4)
    mm.reset_memory(batch_size=1)

    # Identity-fy the memory matrix
    mm.memory_write(_t(1, 0, 0, 0), _t(1, 1, 1, 1), _t(1, 0, 0, 0))
    mm.memory_write(_t(0, 1, 0, 0), _t(1, 1, 1, 1), _t(0, 1, 0, 0))
    mm.memory_write(_t(0, 0, 1, 0), _t(1, 1, 1, 1), _t(0, 0, 1, 0))
    mm.memory_write(_t(0, 0, 0, 1), _t(1, 1, 1, 1), _t(0, 0, 0, 1))

    return mm


class TestAddressing:

    @pytest.mark.parametrize('k, beta, g, shift, gamma, w_prev, expected', [
        (_t(1, 0, 0, 0), _t(100), _t(1), _t(0, 1, 0), _t(100), _t(0, 0, 0, 0), _t(1, 0, 0, 0)), # test similarity/interpolation
    ])
    def test_addressing(self, mem, k, beta, g, shift, gamma, w_prev, expected):
        w = mem.access_memory(k, beta, g, shift, gamma, w_prev)
        assert torch.equal(w.data, expected.data)
