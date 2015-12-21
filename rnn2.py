# -*- coding: UTF-8 -*-
import os
import pickle
import numpy
import theano
import blocks
import math
from blocks.bricks import Tanh
from blocks.algorithms import GradientDescent, Scale, AdaDelta, Momentum, CompositeRule, StepClipping
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
from fuel.transformers import Batch, Padding, Mapping
from fuel.schemes import ConstantScheme
from fuel_test import get_unique_chars

testing = False

num_batches = 30
if not testing:
   pickled_filenames = 'pickled_filenames_1gu.pkl'
   unique_chars = 'charset_1gu.pkl'
   save_path = 'rnn_dump_1gu2.thn'
else:
   pickled_filenames = 'pickled_filenames_toy.pkl'
   unique_chars = 'charset_toy.pkl'
   save_path = 'rnn_dump_toy.thn'
data_path = '/pio/lscratch/1/i246059/language-model/data/plwiki_1gu/'
lstm_dim = 512

def switch_first_two_axes(batch):
   result = []
   for array in batch:
      if array.ndim == 2:
         result.append(array.transpose(1, 0))
      else:
         result.append(array.transpose(1, 0, 2))
   return tuple(result)

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
      dictionary = {'<UNK>' : 0}
      alphabet = u'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
      alphabet += u'1234567890~`!@#$%^&*()-_+=[]{}\|;:.>,</?\n "'
      alphabet += u"'ęóąśłżźćń"
      i = 1
      for letter in alphabet:
         dictionary[letter] = i
         i += 1
      return dictionary
   if testing:
      files = ['data/toy/1', 'data/toy/2', 'data/toy/3']
      pf = open(pickled_filenames, 'w')
      pickle.dump(files, pf)
      pf.close()
   else:
      files = get_training_files()
   files_no = len(files)
   total_size = 0
   for f in files:
      total_size += os.path.getsize(f)
   dictionary = build_dictionary(files)
   # print files
   print dictionary
   print ''.join(dictionary.keys())
   text_files = TextFile(files = files,
                         dictionary = dictionary,
                         bos_token = None,
                         eos_token = None,
                         unk_token = '<UNK>',
                         level = 'character')
   alphabet_size = len(dictionary.keys())
   batch = DataStream(text_files)
   batch = Batch(batch, iteration_scheme=ConstantScheme(num_batches))
   batch = Padding(batch)
   batch = Mapping(batch, switch_first_two_axes)
   
   return batch, alphabet_size, files_no, total_size

def build_generator():
   def build_rnn(lstm_dim):
      lstm1 = LSTM(dim=lstm_dim, use_bias=False,
                  weights_init=Orthogonal())
      lstm2 = LSTM(dim=lstm_dim, use_bias=False,
                  weights_init=Orthogonal())

      return RecurrentStack([lstm1, lstm2],
                           name="transition")


   rnn = build_rnn(lstm_dim)
   readout = Readout(readout_dim = alphabet_size,
                     source_names=["states#1"],
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

print "GO"
x = tensor.lmatrix('features')
# x.tag.test_value = numpy.array([[1,2,3,2,1], [2,3,2,4,4], [1,2,3,2,4]], dtype='int64')
print "X is OK"
mask = tensor.fmatrix('features_mask')
# mask.tag.test_value = numpy.array([[1,1,1,1,1], [1,1,1,0,0], [1,1,1,1,0]], dtype='float32')
print "MASK is OK"
# theano.config.compute_test_value = 'warn'
# print dir(x)
# print x.dtype, mask.dtype
fuel_data, alphabet_size, files_no, signs_no = build_training_set()
print "TEST SET BUILT"

seq_gen = build_generator()
seq_gen.push_initialization_config()
seq_gen.initialize()
print "SEQ_GEN PASSED"

cost_matrix = seq_gen.cost_matrix(x, mask=mask)
# BATCHE X CZAS
# ostatnie 200 to cost_matrix[:,-200:]
# print x.tag.test_value
# print mask.tag.test_value
# print mask.tag.test_value.sum()
# print cost_matrix.tag.test_value.sum()
# print cost_matrix.tag.test_value
cost = math.log(math.e, 2) * aggregation.mean(cost_matrix[:,-200:].sum(), mask[:,-200:].sum())
cost.name = "bits_per_character"
cost_cg = ComputationGraph(cost)

# step_rules = CompositeRule([Momentum(learning_rate=0.02, momentum=0.05), StepClipping(threshold=5.)])
step_rules = CompositeRule([Scale(0.01), StepClipping(threshold=5.)])

algorithm = GradientDescent(
                cost=cost,
                parameters=list(Selector(seq_gen).get_parameters().values()),
                step_rule=step_rules)
# step_rule=Scale(0.001))
# step_rule=AdaDelta())

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
extensions.append(Printing(every_n_batches=100))

main_loop = MainLoop(
                algorithm=algorithm,
                data_stream=fuel_data,
                model=Model(cost),
                extensions=extensions)

main_loop.run()
