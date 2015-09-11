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
from blocks.extensions import FinishAfter, Printing
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
num_batches = 10000


files = ['data/plwiki/art' + str(i) for i in range(1, 500)]
dictionary = get_unique_chars(files)
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
                            transition=rnn,
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

cg_variables = ComputationGraph(cost).variables
print "ZMIENNE W LSTM1"
print VariableFilter(roles=[WEIGHT], bricks=[lstm1])(cg_variables)
print "ZMIENNE W LSTM2"
print VariableFilter(roles=[WEIGHT], bricks=[lstm2])(cg_variables)
print "ZMIENNE W LSTM3"
print VariableFilter(roles=[WEIGHT], bricks=[lstm3])(cg_variables)
print "ZMIENNE RAZEM W RNN"
print VariableFilter(roles=[WEIGHT], bricks=[rnn])(cg_variables)

# theano.printing.pydotprint(cost, outfile="./pics/symbolic_graph_unopt.png", var_with_name_simple=True)

variables = VariableFilter(roles=[WEIGHT], bricks=[lstm1])(cg_variables)
for var in variables:
    print var.owner


algorithm = GradientDescent(
                cost=cost,
                parameters=list(Selector(seq_gen).get_parameters().values()),
                step_rule=Scale(0.001))

main_loop = MainLoop(
                algorithm=algorithm,
                data_stream=DataStream(text_files),
                model=Model(cost),
                extensions=[Checkpoint(save_path, every_n_batches=500),
                            Printing(every_n_batches=100),
                            TrainingDataMonitoring([cost], prefix="this_step",
                                                   after_batch=True),
                            TrainingDataMonitoring([cost], prefix="average",
                                                   every_n_batches=100),
                            FinishAfter(after_n_batches=num_batches)])

main_loop.run()

