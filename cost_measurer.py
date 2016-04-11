# -*- coding: UTF-8 -*-
import numpy as np
import sys
import yaml
import theano
from theano import tensor
from data import build_datasets
from model import build_model, build_algorithm
from monitor import build_extensions
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.extensions.saveload import Load
import cPickle as pickle
from blocks.graph import ComputationGraph

class CostMeasurer:
    def __init__(self, config_dict):
        print config_dict
        train, valid, alphabet = build_datasets(config_dict)
        generator, cost = build_model(len(alphabet), config_dict)
        algorithm = build_algorithm(generator, cost, config_dict)
        extensions = build_extensions(cost, algorithm, valid, config_dict)
        main_loop = MainLoop(algorithm=algorithm, data_stream=train,
                             model=Model(cost), extensions=extensions)
        ml = Load(config_dict['checkpoint_path'], load_log=True)
        ml.load_to(main_loop)
        generator = main_loop.model.get_top_bricks()[-1]
        
        self.numbers_from_text = pickle.load(open(config_dict['dict_path']))

        x = tensor.lmatrix('sample')
        cost_cg = generator.cost(x)
        self.cost_f = theano.function([x], cost_cg)

    def tokenise(self, sentence):
        result = []
        sentence_stack = ''
        for char in sentence.strip():
            sentence_stack += char
            if sentence_stack in self.numbers_from_text:
                result.append(self.numbers_from_text[sentence_stack])
                sentence_stack = ''
        return result

    def cost(self, sequence):
        numbers = np.array(self.tokenise(sequence), dtype=np.int64)
        return self.cost_f(numbers.reshape(numbers.shape[0], 1))         

    def rate_file(self, file_path):
        f = open(file_path)
        results = {}
        for (i, line) in enumerate(f):
            results[i + 1] = (line[:-1],self.cost(line[:-1]))
        f.close()
        return results

if __name__ == '__main__':
    config_dict = yaml.load(open(sys.argv[1], 'r'))
    cm = CostMeasurer(config_dict)
    print cm.numbers_from_text
    print cm.rate_file('test')
    while True:
        seq = raw_input()
        print 'Cost for ' + seq + ':'
        print cm.cost(seq)
