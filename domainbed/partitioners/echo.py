from domainbed.partitioners.partitioner import Partitioner
from domainbed.partitioners.partition import Partition

class EchoPartitioner(Partitioner):
    """
    Mock that only repeats the original partition.
    """
    def __init__(self, hparams):
        super(EchoPartitioner, self).__init__(hparams)

    def split(self, dataset):
        self.partition = Partition()
        for old_env, values in enumerate(dataset):
            for i in range(len(values)):
                self.partition.assign(original_environment=old_env, old_index=i, new_environment=old_env)
        return self.partition