import torch

class Partitioner(torch.nn.Module):
    """
    A subclass of Partitioner implements a partitioner.
    Subclasses should implement the following:
    - split()
    """
    def __init__(self, hparams):
        super(Partitioner, self).__init__()
        self.hparams = hparams

    def split(self, x):
        raise NotImplementedError