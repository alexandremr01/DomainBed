import torch

from domainbed.partitioners.partition import Partition

class Partitioner(torch.nn.Module):
    """
    A subclass of Partitioner implements a partitioner.
    Subclasses should implement the following:
    - split()
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Partitioner, self).__init__()
        self.hparams = hparams
        self.partition = Partition()

    def split(self, x):
        raise NotImplementedError