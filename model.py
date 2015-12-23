import theano
import blocks
import math
from theano import tensor
from blocks.bricks.recurrent import LSTM, RecurrentStack
from blocks.bricks.sequence_generators import (SequenceGenerator,
        Readout, SoftmaxEmitter, LookupFeedback)
from blocks.initialization imort Uniform


def build_model(alphabet_size, config):
    layers = config['lstm_layers']
    dimensions = [config['lstm_dim_' + str(i)] for i in range(layers)]
    uniform_width = config['lstm_init_width']
    stack = []
    for dim in dimensions:
        stack.append(LSTM(dim=dim, use_bias=False, 
                          weights_init = Uniform(width=uniform_width)))
    recurrent_stack = RecurrentStack(stack, name='transition')

    readout = Readout(readout_dim=alphabet_size,
                      source_names=['states#' + str(layers - 1)],
                      emitter=SoftmaxEmitter(name='emitter'),
                      feedback_brick=LookupFeedback(alphabet_size,
                                                    feedback_dim=alphabet_size,
                                                    name='feedback'),
                      name='readout')

    generator = SequenceGenerator(readout=readout,
                                  transition=recurrent_stack,
                                  weights_init=Uniform(width=uniform_width),
                                  biases_init=Constant(0),
                                  name='generator')
    generator.push_initialization_config()
    generator.initialize()

    x = tensor.lmatrix('features')
    mask = tensor.fmatrix('features_mask')
    cost_matrix = generator.cost_matrix(x, mask=mask)

    length = config['batch_length'] - config['batch_overlap']

    log2e = math.log(math.e, 2)
    cost = log2e * aggregation.mean(cost_matrix[:,-length:].sum(), 
                                    mask[:-length:].sum())

    return cost
