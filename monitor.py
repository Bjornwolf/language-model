import numpy as np
from blocks.graph import ComputationGraph
from blocks.extensions import Printing, Timing
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import (TrainingDataMonitoring, 
        DataStreamMonitoring)
from blocks.extensions.training import SharedVariableModifier


class VarModifier:
    def __init__(self, c):
        self.c = np.float32(c)

    def __call__(self, x, y):
        return np.float32(self.c * y)

def build_extensions(cost, algorithm, valid, config):
    cost_cg = ComputationGraph(cost)
    observables = list(cost_cg.outputs)
    observables.append(algorithm.total_step_norm)
    observables.append(algorithm.total_gradient_norm)

    average = config['average_frequency']
    checkpoint_path = config['checkpoint_path']
    checkpoint_frequency = config['checkpoint_frequency']
    printing_frequency = config['printing_frequency']
    valid_frequency = config['valid_frequency']

    i = 0
    while config['step_rule_'+ str(i)] != 'RMSProp':
        i += 1

    observables.append(algorithm.step_rule.components[i].learning_rate)
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
                                           every_n_batches=valid_frequency))
    extensions.append(Checkpoint(checkpoint_path,
                                 every_n_batches=checkpoint_frequency, 
                                 use_cpickle=True))
    extensions.append(Printing(every_n_batches=printing_frequency))
    if 'lr_decay' in config:
        inst = VarModifier(config['lr_decay'])
        extensions.append(SharedVariableModifier(algorithm.step_rule.components[i].learning_rate,
                                                  inst, 2, every_n_batches=config['lr_decay_frequency'], after_batch=False))

    return extensions
