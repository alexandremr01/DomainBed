import numpy as np
from domainbed.partitioners.partitioner import Partitioner
from domainbed.partitioners.partition import Partition

import torch
from torch import optim

class Decorr(Partitioner):
    """
    Implementation of the Decorr algorithm, as seen in Decorr: Environment Partitioning
    for Invariant Learning and OOD Generalization, Liao et al. 2022.
    """
    def __init__(self, input_shape, hparams):
        super(Decorr, self).__init__(hparams)
        self.regularization_weight = hparams['regularization_weight']
        self.max_epochs = hparams['max_epochs']
        self.num_environments = hparams['num_environments']
        self.lr = hparams['lr']
        self.p0 = hparams['p0']
        self.input_shape = input_shape
        if torch.cuda.is_available():
          self.device = 'cuda'
        else:
          self.device = 'cpu'

    def split(self, dataset):
        self.partition = Partition(dataset)
        # remaining_indexes stores pairs of (env, index) of the original dataset
        remaining_indexes = set([(old_env, i) for old_env, values in enumerate(dataset) for i in range(len(values))])

        for j in range(0, self.num_environments-1):
            # generates random numbers in the range [p0, 1)
            num_remaining = len(remaining_indexes)
            X_r = torch.tensor(np.array([dataset[env][i] for (env, i) in remaining_indexes])).to(self.device)
            env_w = (torch.randn(num_remaining).double().to(self.device) * (1-self.p0) + self.p0).requires_grad_()
            optimizer = optim.Adam([env_w], lr=self.lr)
            for epoch in range(self.max_epochs):
                real_w = env_w.sigmoid()
                l = self._distance_to_identity(X_r, real_w) 
                l += self.regularization_weight * (real_w.sum()/num_remaining - 1 / (self.num_environments-j))**2

                optimizer.zero_grad()
                l.backward()
                optimizer.step()

                if epoch % 100 == 0:
                    print(f'epoch {epoch+1}: loss = {l.item():.7f}')

            removing_indexes = [ ]
            probs = env_w.sigmoid()
            for i, ix in enumerate(remaining_indexes):
                r = np.random.random()
                if r < probs[i]:
                    self.partition.assign(original_environment=ix[0], old_index=ix[1], new_environment=j)
                    removing_indexes.append(ix)
            for ix in removing_indexes:
                remaining_indexes.remove(ix)
        for i, ix in enumerate(remaining_indexes):
            self.partition.assign(original_environment=ix[0], old_index=ix[1], new_environment=self.num_environments-1)
        return self.partition

    def _distance_to_identity(self, X_r, w):
        feature_dimensions = self.input_shape
        R = torch.zeros((feature_dimensions, feature_dimensions)).to(self.device)
        for i in range(feature_dimensions):
            for j in range(feature_dimensions):
                R[i, j] = self._feature_correlation(X_r, i, j, w)
        return torch.linalg.norm( R - torch.eye(feature_dimensions).to(self.device) )
    
    def _feature_correlation(self, X, i, j, w):
        w_norm = w / w.sum()
        Xw = torch.dot(X[:, i], w_norm)
        Yw = torch.dot(X[:, j], w_norm)
        diffX = X[:, i]-Xw
        diffY = X[:, j]-Yw

        num = w_norm.dot(diffX.multiply(diffY))
        den = torch.sqrt(w_norm.dot(torch.pow(diffX, 2))) * torch.sqrt(w_norm.dot(torch.pow(diffY, 2)))
        return num / den


