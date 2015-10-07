import os
import pickle
import numpy
import theano
import blocks
from blocks.bricks import Tanh
from blocks.algorithms import GradientDescent, Scale
from blocks.bricks.recurrent import LSTM
from blocks.bricks.recurrent import RecurrentStack
from blocks.bricks.sequence_generators import (SequenceGenerator, 
        Readout, SoftmaxEmitter, LookupFeedback)
from blocks.initialization import Orthogonal, IsotropicGaussian, Constant
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks.graph import ComputationGraph
from blocks.extensions import FinishAfter, Printing, Timing
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.main_loop import MainLoop
from blocks.select import Selector
# <DEBUG>
from blocks.filter import VariableFilter
from blocks.roles import WEIGHT
# </DEBUG>
from theano import tensor
from fuel.datasets import Dataset, TextFile
from fuel.streams import DataStream
from fuel_test import get_unique_chars

#TODO rozszerzenie recurrentstack na reset stanu
#TODO przerobic na klase
save_path = 'rnn_dump.thn'
num_batches = 2000000
pickled_filenames = 'pickled_filenames.pkl'
unique_chars = 'charset.pkl'

if os.path.isfile(pickled_filenames):
   pf = open(pickled_filenames, 'r')
   files = pickle.load(pf)
   pf.close()
else:
   files = []
   data_location = 'data/plwiki/'
   for (dirname, _, filenames) in os.walk(data_location):
      files += map(lambda x: dirname + '/' + x, filenames)
   os.system('touch ' + pickled_filenames)
   pf = open(pickled_filenames, 'w')
   pickle.dump(files, pf)
   pf.close()

print len(files), ' files on the list'

# files = ['data/plwiki/art' + str(i) for i in range(1, 500)]
if os.path.isfile(unique_chars):
   uc = open(unique_chars, 'r')
   dictionary = pickle.load(uc)
   uc.close()
else:
   dictionary = get_unique_chars(files)
   os.system('touch ' + unique_chars)
   uc = open(unique_chars, 'w')
   pickle.dump(dictionary, uc)
   uc.close()
print dictionary
text_files = TextFile(files = files,
                      dictionary = dictionary,
                      bos_token = None,
                      eos_token = None,
                      unk_token = '<UNK>',
                      level = 'character')
alphabet_size = len(dictionary.keys())

lstm_dim = 512

lstm1 = LSTM(dim=lstm_dim, use_bias=False,
            weights_init=Orthogonal())
lstm2 = LSTM(dim=lstm_dim, use_bias=False,
            weights_init=Orthogonal())
lstm3 = LSTM(dim=lstm_dim, use_bias=False,
            weights_init=Orthogonal())

rnn = RecurrentStack([lstm1, lstm2, lstm3],
                     name="transition")

readout = Readout(readout_dim = alphabet_size,
                  source_names=["states"],
                  emitter=SoftmaxEmitter(name="emitter"),
                  feedback_brick=LookupFeedback(alphabet_size,
                                                feedback_dim=alphabet_size,
                                                name="feedback"),
                  name="readout")

seq_gen = SequenceGenerator(readout=readout,
                            transition=lstm1,
                            weights_init=IsotropicGaussian(0.01),
                            biases_init=Constant(0),
                            name="generator")

seq_gen.push_initialization_config()
rnn.weights_init = Orthogonal()
seq_gen.initialize()

# z markov_tutorial
x = tensor.lvector('features')
x = x.reshape( (x.shape[0], 1) )
cost = aggregation.mean(seq_gen.cost_matrix(x[:,:]).sum(), x.shape[1])
cost.name = "sequence_log_likelihood"
cost_cg = ComputationGraph(cost)

# theano.printing.pydotprint(cost, outfile="./pics/symbolic_graph_unopt.png", var_with_name_simple=True)

algorithm = GradientDescent(
                cost=cost,
                parameters=list(Selector(seq_gen).get_parameters().values()),
                step_rule=Scale(0.001))

# AUDIOSCOPE OBSERVABLES (some)
observables = []
observables += cost_cg.outputs
observables.append(algorithm.total_step_norm)
observables.append(algorithm.total_gradient_norm)

print observables

# AUDIOSCOPE EXTENSIONS
extensions = []
extensions.append(Timing(after_batch=True))
extensions.append(TrainingDataMonitoring(list(observables), after_batch=True))
averaging_frequency = 1000
average_monitor = TrainingDataMonitoring(observables, prefix="average", every_n_batches=averaging_frequency)
extensions.append(average_monitor)
checkpointer = Checkpoint(save_path, every_n_batches=500, use_cpickle=True)
extensions.append(checkpointer)
extensions.append(Printing(every_n_batches=10))



main_loop = MainLoop(
                algorithm=algorithm,
                data_stream=DataStream(text_files),
                model=Model(cost),
                extensions=extensions)

main_loop.run()

