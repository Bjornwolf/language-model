import theano
import blocks
import math
from theano import tensor
from blocks.bricks.recurrent import LSTM, RecurrentStack
from blocks.bricks.sequence_generators import (SequenceGenerator,
        Readout, SoftmaxEmitter, LookupFeedback)
from blocks.initialization import Uniform, Constant
from blocks.select import Selector
from blocks.algorithms import (GradientDescent, Scale, Momentum, AdaDelta, 
        RMSProp, StepClipping, Adam, CompositeRule)
from blocks.monitoring import aggregation

def build_model(alphabet_size, config):
    layers = config['lstm_layers']
    dimensions = [config['lstm_dim_' + str(i)] for i in range(layers)]
    uniform_width = config['lstm_init_width']
    stack = []
    for dim in dimensions:
        stack.append(LSTM(dim=dim, use_bias=True, 
                          weights_init = Uniform(width=uniform_width),
                          forget_init=Constant(1.)))
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

    log2e = math.log(math.e, 2)
    if 'batch_length' in config:
        length = config['batch_length'] - config['batch_overlap']

        cost = log2e * aggregation.mean(cost_matrix[:,-length:].sum(), 
                                    mask[:,-length:].sum())
    else:
        cost = log2e * aggregation.mean(cost_matrix[:,:].sum(), 
                                    mask[:,:].sum())
        
    cost.name = 'bits_per_character'

    return generator, cost


def build_rule(rule_type, params):
    if rule_type == 'Scale':
        return Scale(learning_rate=params['learning_rate'])
    if rule_type == 'Momentum':
        return Momentum(learning_rate=params['learning_rate'], 
                        momentum=params['momentum'])
    if rule_type == 'AdaDelta':
        return AdaDelta(decay_rate=params['decay_rate'], 
                        epsilon=params['epsilon'])
    if rule_type == 'RMSProp':
        return RMSProp(learning_rate=params['learning_rate'],
                       decay_rate=params['decay_rate'],
                       max_scaling=params['max_scaling'])
    if rule_type == 'StepClipping':
        return StepClipping(threshold=params['threshold'])
    if rule_type == 'Adam':
        return Adam(learning_rate=params['learning_rate'],
                    beta1=params['beta1'], beta2=params['beta2'],
                    epsilon=params['epsilon'], 
                    decay_factor=params['decay_factor'])


def build_algorithm(generator, cost, config):
    rules_no = config['step_rules_no']
    if rules_no == 1:
        rule_type = config['step_rule']
        rule_params = config['step_rule_params']
        rule = build_rule(rule_type, rule_params)
    else:
        rules = []
        for i in range(rules_no):
            rule_type = config['step_rule_' + str(i)]
            rule_params = config['step_rule_' + str(i) + '_params']
            rules.append(build_rule(rule_type, rule_params))
            rule = CompositeRule(rules)

    parameters=list(Selector(generator).get_parameters().values())
    algorithm = GradientDescent(cost=cost, 
                                parameters=parameters, 
                                step_rule=rule)
    return algorithm






