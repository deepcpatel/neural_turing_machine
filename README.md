# Neural Turing Machine implementation in PyTorch
An implementation of [Neural Turing Machine](https://arxiv.org/pdf/1410.5401.pdf) in PyTorch based on [loudinthecloud's](https://github.com/loudinthecloud) PyTorch implementation

**Original Platform:** Ubuntu 18.04</br>
**Language:** Python 3.6</br>
**Libraries required:** (1). NumPy </br>
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;(2). PyTorch

Run the code by writing following in the terminal:</br>
``python3 train.py``

**Note-1:** There are two versions og NTM here - Original(NTM) and stable(NTM\_stable). The stable version is the adaptation of original version with changes suggested in [Implementing Neural Turing Machines](https://arxiv.org/pdf/1807.08518.pdf) paper. To use either version, just comment or uncomment the imported packages in ``tasks/copy_task.py`` file.</br></br>
**Note-2:** The code is not extensively tested due to the lack of powerful hardware.

## Reference
1). https://github.com/loudinthecloud/pytorch-ntm </br>
2). https://arxiv.org/pdf/1410.5401.pdf </br>
3). https://arxiv.org/pdf/1807.08518.pdf
