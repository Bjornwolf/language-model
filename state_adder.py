from cost_measurer import CostMeasurer
import numpy as np
import yaml
import sys
from blocks.utils import dict_union, dict_subset
import theano
from theano import tensor
import zipfile

dict_name = sys.argv[1]

cm = CostMeasurer(yaml.load(open(dict_name)))
seq_gen = cm.main_loop.model.get_top_bricks()[0]


input_seq = tensor.lmatrix('x')

batch_size = 1
feedback = seq_gen.readout.feedback(input_seq)
inputs = seq_gen.fork.apply(feedback, as_dict=True)
results = seq_gen.transition.apply(
    mask=None, return_initial_states=False, as_dict=True,
    **dict_union(inputs, {}, {}))
    # **dict_union(inputs, seq_gen._state_names, seq_gen._context_names))

states = {name: results[name] for name in seq_gen._state_names}
get_states = theano.function([input_seq], states)
example_in = np.array(cm.tokenise('<d><u><p><a>>'))
new_states = get_states(example_in.reshape(example_in.shape[0], 1))


bootup_seq = ''.join(map(lambda x: x[:-1], open(sys.argv[2]).readlines()))

print cm.cost(bootup_seq)

states_list = new_states.keys()

# for (i, key) in enumerate(par_list):
#     all_param_dict[key] = np.mean(new_states[states_list[i]], axis=0).reshape(all_param_dict[key].shape)

mod = cm.main_loop.model
parameters = filter(lambda x: 'initial' in x, mod._parameter_dict.keys())

print parameters
print states_list

for param in parameters:
    if 'state' in param:
        name = 'states'
    else:
        name = 'cells'
    if '#0' not in param:
        name += param[param.find('#'):param.find('.')]
    
    new_value = np.mean(new_states[name], axis=0)
    val_shape = mod._parameter_dict[param].get_value().shape
    mod._parameter_dict[param].set_value(new_value.reshape(val_shape))
print cm.cost(bootup_seq)

import IPython
IPython.embed()
