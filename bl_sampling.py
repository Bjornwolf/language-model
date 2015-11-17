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
from blocks.serialization import load
from fuel_test import get_unique_chars

charset = pickle.load(open('charset_bl.pkl', 'rb'))
charset = {v:k for k,v in charset.iteritems()}

main_loop = load(open("models/rnn_bkp_bl.thn", "rb"))
generator = main_loop.model.get_top_bricks()[-1]

sampler = ComputationGraph(generator.generate(
    n_steps=1000, batch_size=10, iterate=True)).get_theano_function()

samples = sampler()

outputs = samples[-2]

for i in xrange(outputs.shape[1]):
   print "Sample number ", i, ": ",
   print ''.join(map(lambda x: charset[x], outputs[:,i]))

