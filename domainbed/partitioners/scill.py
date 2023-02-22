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

class SCILL(Partitioner):
    """
    Implementation of the SCILL algorithm, as seen in When Does Group Invariant Learning Survive
    Spurious Correlations?, Chen et al. 2022.
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Decorr, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.threshold = hparams.threshold

    def split(self, dataset):
        self.blocking_multi(indexes=, probs=, labels=, thr=self.threshold)

    def blocking_multi(self, indexes, probs, labels, thr, use_clustering=True, method='anova'):
        queue = deque([indexes])
        blocks = []
        split_count = 0
        # old_split_idx = 0
        n_labels = len(np.unique(labels))
        ulabels = np.arange(n_labels).astype(int)
        while queue:
            cur_indexes = queue.popleft().astype(int)
            cur_probs = probs[cur_indexes]
            cur_labels = labels[cur_indexes]
            if len(np.unique(cur_labels)) < 2:
                blocks.append((cur_probs, 1., cur_indexes))
                continue
            f_stats, pvalues = check_balancing(cur_probs, cur_labels, n_labels, method)
            if f_stats > thr and split_count < num_splits:
                if use_clustering:
                    cur_ulabels = np.unique(cur_labels)
                    cluster_vecs = cur_probs
                    n_clusters = 2
                    cluster_func = KMeans(n_clusters, init='k-means++', max_iter=10000).fit(cluster_vecs)
                    cluster_ids = cluster_func.labels_
                    for i in range(n_clusters):
                        queue.append(cur_indexes[cluster_ids==i])
                else:
                    label = ulabels[pvalues < thr][0][0]
                    split_dim = int(label)
                    split_probs = np.array([prob[split_dim] for prob in cur_probs])
                    median = np.median(split_probs)
                    sub1_indexes = cur_indexes[split_probs <= median]
                    sub2_indexes = cur_indexes[split_probs > median]
                    queue.append(sub1_indexes)
                    queue.append(sub2_indexes)
                split_count += 1
            else:
                blocks.append((cur_probs, pvalues, cur_indexes))
        return blocks


    def check_balancing(self, probs, labels, n_labels, method='anova'):
        unique_labels = np.unique(labels)
        input_args = [] # #labels arrays, each of num_samples * dims
        pvalues = []
        if len(unique_labels) == 2:
            if n_labels == 2:
                linear_p = np.array([np.log(prob[0] / prob[1]) for prob in probs])
                for i in range(2): 
                    input_args.append(linear_p[labels == i])
            else:
                for label in unique_labels:
                    input_args.append(probs[labels == label])
            t_statistic, pvalues = ttest_ind(*input_args)
            return t_statistic, np.array(pvalues)
        else:
            if method == 'krustal':
                pvalues = []
                for i in range(n_labels - 1):
                    input_args = []
                    for label in unique_labels:
                        if (labels == label).sum() < 100:
                            continue
                        input_args.append(probs[labels == label][:,i])
                    if len(input_args) < 2:
                        return 0, np.ones(2)
                    statistic, pvalue = kruskal(*input_args)
                    pvalues.append(pvalue)
            if method == 'anova':
                for label in unique_labels:
                    if (labels == label).sum() < 500:
                        continue
                    probs_label = probs[labels == label]
                    input_args.append(probs_label[:,:-1])
                if len(input_args) < 2:
                    return 0, np.ones(2)
                statistic, pvalues = f_oneway(*input_args)
            return statistic, np.array(pvalues)