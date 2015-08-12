import numpy
import theano
import blocks
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
from theano import tensor
from fuel_test import get_unique_chars

#TODO
# alphabet_size

alphabet_size = 40
lstm_dim = 512

lstm1 = LSTM(dim=lstm_dim, use_bias=False,
            weights_init=Orthogonal())
lstm2 = LSTM(dim=lstm_dim, use_bias=False,
            weights_init=Orthogonal())
lstm3 = LSTM(dim=lstm_dim, use_bias=False,
            weights_init=Orthogonal())
rnn = RecurrentStack([lstm1, lstm2, lstm3], name="transition")

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

#TODO rozszerzenie recurrentstack na reset stanu
#TODO zobaczyc, jak wygladaja batche i zrobic to samo z jakims tekstem (fuel)
#TODO dane mam z polskiej wiki, XML, <mediawiki> <page> <text>
#TODO przerobic na klase

# z markov_tutorial
x = tensor.lmatrix('data')
cost = aggregation.mean(seq_gen.cost_matrix(x[:,:]).sum(), x.shape[1])
cost.name = "sequence_log_likelihood"

algorithm = GradientDescent(
                cost=cost,
                parameters=list(Selector(seq_gen).get_parameters().values()),
                step_rule=Scale(0.001))
files = ['data/plwiki/art' + str(i) for i in range(1, 500)]
text_files = TextFile(files = files,
                      dictionary = get_unique_chars(files),
                      bos_token = None,
                      eos_token = None,
                      unk_token = '<UNK>',
                      level = 'character')

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

