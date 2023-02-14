from domainbed.partitioners.echo import EchoPartitioner
from domainbed.partitioners.decorr import Decorr

PARTITIONERS = {
    'Echo': EchoPartitioner,
}
def get_partitioner_class(algorithm_name):
    """Return the partitioner class with the given name."""
    if algorithm_name not in PARTITIONERS:
        raise NotImplementedError("Partitioner not found: {}".format(algorithm_name))
    return PARTITIONERS[algorithm_name]