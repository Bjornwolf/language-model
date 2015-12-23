from blocks.graph import ComputationGraph
from blocks.extensions import Printing, Timing
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import (TrainingDataMonitoring, 
        DataStreamMonitoring)


def build_extensions(cost, algorithm, config):
    cost_cg = ComputationGraph(cost)
    observables = list(cost_cg.outputs)
    observables.append(algorithm.total_step_norm)
    observables.append(algorithm.total_gradient_norm)

    average = config['average_frequency']
    checkpoint_path = config['checkpoint_path']
    checkpoint_frequency = config['checkpoint_frequency']
    printing_frequency = config['printing_frequency']
    valid_frequency = config['valid_frequency']

    extensions = []
    extensions.append(Timing(after_batch=True))
    extensions.append(TrainingDataMonitoring(observables, 
                                             after_batch=True))
    extensions.append(TrainingDataMonitoring(observables, prefix='average',
                                             every_n_batches=average))
    extensions.append(DataStreamMonitoring(observables, valid, prefix='valid',
                                           before_first_epoch=False,
                                           after_epoch=True,
                                           after_training=False,
                                           every_n_batches=valid_frequency)
    extensions.append(Checkpoint(checkpoint_path,
                                 every_n_batches=checkpoint_frequency, 
                                 use_cpickle=True))
    extensions.append(Printing(every_n_batches=printing_frequency))

    return extensions
