import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import copy
import numpy as np
from collections import OrderedDict

from domainbed import networks
from domainbed.lib.misc import (
    random_pairs_of_minibatches, split_meta_train_test, ParamDict,
    MovingAverage, l2_between_dicts, proj
)

class Partition:
    """Describes one way to partition the dataset."""
    def __init__(self) -> None:
        self.mapping = { }
        self.new_environments = set( )

    def assign(self, original_environment, old_index, new_environment):
        if original_environment not in self.mapping:
            self.mapping[original_environment] = { }
        self.mapping[original_environment][old_index] = new_environment
        self.new_environments.add(new_environment)

    def get_mapping_original_to_new(self):
        """This format is better to make assignments based on the original dataset. It returns a map in the format
        'old_environment': {'index': 'new_environment'}  """
        return self.mapping

    def get_mapping_new_to_original(self):
        """This format is better to analyze the composition of new environments."""
        new_to_original = { }
        for env in self.new_environments:
            new_to_original[env] = { }
        for original_env, elements in self.mapping.items():
            for ix, new_environment in elements.items():
                if new_to_original[new_environment].get(original_env) is None:
                    new_to_original[new_environment][original_env] = [ ]
                new_to_original[new_environment][original_env].append(ix)
        return new_to_original



def get_partitioner_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Partitioner not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

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

class EchoPartitioner(Partitioner):
    """
    Mock that only repeats the original partition.
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(EchoPartitioner, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

    def split(self, dataset):
        for old_env, values in enumerate(dataset):
            for i in range(len(values)):
                self.partition.assign(original_environment=old_env, old_index=i, new_environment=old_env)
        return self.partition
