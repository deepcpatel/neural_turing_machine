# Neural Turing Machine implementation in PyTorch
An implementation of [Neural Turing Machine](https://arxiv.org/pdf/1410.5401.pdf) [2] in PyTorch with reference from [loudinthecloud's](https://github.com/loudinthecloud) PyTorch implementation [1].

**Original Platform:** Ubuntu 18.04</br>
**Language:** Python 3.6</br>
**Libraries required:** NumPy, PyTorch 1.1.0

Run the code by writing following in the terminal:</br>
``python3 train.py``

**Note-1:** There are two versions of NTM here - original(NTM) and stable(NTM\_stable). The stable version is the adaptation of original version with modifications suggested in [Implementing Neural Turing Machines](https://arxiv.org/pdf/1807.08518.pdf) [3] paper. To use either version, just comment or uncomment the imported packages in ``tasks/copy_task.py`` file.</br></br>
**Note-2:** The code is not extensively tested. However, you are welcome to report any bugs if found.</br></br>
**Note-3:** You are encouraged to contribute to this repo or provide any suggestions. 

## Reference
**[1]**. https://github.com/loudinthecloud/pytorch-ntm </br>
**[2]**. Graves, A., Wayne, G. and Danihelka, I., 2014. Neural turing machines. arXiv preprint arXiv:1410.5401. </br>
**[3]**. Collier, M. and Beel, J., 2018, October. Implementing neural turing machines. In International Conference on Artificial Neural Networks (pp. 94-104). Springer, Cham.
