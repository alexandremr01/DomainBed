from domainbed.partitioners.partitioner import Partitioner

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