import numpy
import theano
import blocks
from blocks.bricks import Initializable
from blocks.bricks.recurrent import BaseRecurrent, LSTM


class LSTMStack(BaseRecurrent, Initializable):
   def __init__(self, layers_no, dim, alphabet_size, batch_size):
      # characters -> 1-of-N embedder -> N-to-dim -> LSTM#0 -> ... -> LSTM#(layers_no-1) -> dim-to-N -> softmax
      # TODO zdefiniowac blad

      # TODO first_resizer

      # LSTM stack
      self.stack = []
      lstms = map(lambda _: LSTM(dim=dim), range(layers_no))
      for lstm in lstms:
         state, cell = lstm.initial_states(batch_size)
         self.stack.append(lstm, state, cell)
      
      # TODO last_resiser


      # TODO softmax

   def apply(self, text_input):
      resized_input = text_input # TODO first_resizer
      for (i, (lstm, state, cell)) in self.stack:
         
      
