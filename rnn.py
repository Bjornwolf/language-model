import os
import pickle
import numpy
import theano
import blocks
import math
from blocks.bricks import Tanh
from blocks.algorithms import GradientDescent, Scale, AdaDelta
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
from theano import tensor
from fuel.datasets import Dataset, TextFile
from fuel.streams import DataStream
from fuel_test import get_unique_chars

# TODO perplexity
save_path = 'rnn_dump_1g.thn'
num_batches = 2000000
pickled_filenames = 'pickled_filenames_1g.pkl'
unique_chars = 'charset_1g.pkl'
data_path = '/pio/lscratch/1/i246059/language-model/data/plwiki_1g/'
lstm_dim = 512

def build_training_set():
   def get_training_files():
      if os.path.isfile(pickled_filenames):
         pf = open(pickled_filenames, 'r')
         files = pickle.load(pf)
         pf.close()
      else:
         files = []
         data_location = data_path + 'train/'
         for (dirname, _, filenames) in os.walk(data_location):
            files += map(lambda x: dirname + '/' + x, filenames)
         os.system('touch ' + pickled_filenames)
         pf = open(pickled_filenames, 'w')
         pickle.dump(files, pf)
         pf.close()
      return files

   def build_dictionary(files):
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
      return dictionary

   files = get_training_files()
   files_no = len(files)
   total_size = 0
   for f in files:
      total_size += os.path.getsize(f)
   dictionary = build_dictionary(files)
   text_files = TextFile(files = files,
                         dictionary = dictionary,
                         bos_token = None,
                         eos_token = None,
                         unk_token = '<UNK>',
                         level = 'character')
   alphabet_size = len(dictionary.keys())
   return text_files, alphabet_size, files_no, total_size

def build_generator():
   def build_rnn(lstm_dim):
      lstm1 = LSTM(dim=lstm_dim, use_bias=False,
                  weights_init=Orthogonal())
      lstm2 = LSTM(dim=lstm_dim, use_bias=False,
                  weights_init=Orthogonal())
      lstm3 = LSTM(dim=lstm_dim, use_bias=False,
                  weights_init=Orthogonal())

      return RecurrentStack([lstm1, lstm2, lstm3],
                           name="transition")


   rnn = build_rnn(lstm_dim)
   readout = Readout(readout_dim = alphabet_size,
                     source_names=["states#2"],
                     emitter=SoftmaxEmitter(name="emitter"),
                     feedback_brick=LookupFeedback(alphabet_size,
                                                   feedback_dim=alphabet_size,
                                                   name="feedback"),
                     name="readout")
   rnn.weights_init = Orthogonal()

   return SequenceGenerator(readout=readout,
                            transition=rnn,
                            weights_init=IsotropicGaussian(0.01),
                            biases_init=Constant(0),
                            name="generator")

fuel_data, alphabet_size, files_no, signs_no = build_training_set()

seq_gen = build_generator()

seq_gen.push_initialization_config()
seq_gen.initialize()

x = tensor.lvector('features')
x = x.reshape( (x.shape[0], 1) )
cost = math.log(math.e, 2) * aggregation.mean(seq_gen.cost_matrix(x[:,:]).sum(), x.shape[1])
cost.name = "bits_per_character"
cost_cg = ComputationGraph(cost)

algorithm = GradientDescent(
                cost=cost,
                parameters=list(Selector(seq_gen).get_parameters().values()),
                step_rule=AdaDelta())

observables = []
observables += cost_cg.outputs
observables.append(algorithm.total_step_norm)
observables.append(algorithm.total_gradient_norm)

print "OBSERVABLES"
print observables

# 10. dodaj podgladanie zmiennych
extensions = []
# mierzenie czasu
extensions.append(Timing(after_batch=True))
# wypisywanie observables
extensions.append(TrainingDataMonitoring(list(observables), after_batch=True))
# srednie z observables
averaging_frequency = 1000
average_monitor = TrainingDataMonitoring(observables, prefix="average", every_n_batches=averaging_frequency)
extensions.append(average_monitor)
checkpointer = Checkpoint(save_path, every_n_batches=500, use_cpickle=True)
extensions.append(checkpointer)
extensions.append(Printing(every_n_batches=50))

main_loop = MainLoop(
                algorithm=algorithm,
                data_stream=DataStream(fuel_data),
                model=Model(cost),
                extensions=extensions)

main_loop.run()
