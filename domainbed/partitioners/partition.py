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
from torch import optim

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
