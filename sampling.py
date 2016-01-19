# -*- coding: UTF-8 -*-
import sys
import yaml
from data import build_datasets
from model import build_model, build_algorithm
from monitor import build_extensions
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.extensions.saveload import Load
import cPickle as pickle
from blocks.graph import ComputationGraph

config_dict = yaml.load(open(sys.argv[1], 'r'))
print config_dict

train, valid, alphabet = build_datasets(config_dict)
generator, cost = build_model(len(alphabet), config_dict)
algorithm = build_algorithm(generator, cost, config_dict)
extensions = build_extensions(cost, algorithm, valid, config_dict)
main_loop = MainLoop(algorithm=algorithm, data_stream=train,
                     model=Model(cost), extensions=extensions)
ml = Load(config_dict['checkpoint_path'], load_log=True)
print dir(ml)

ml.load_to(main_loop)
generator = main_loop.model.get_top_bricks()[-1]

sampler = ComputationGraph(generator.generate(
    n_steps=1000, batch_size=10, iterate=True)).get_theano_function()

samples = sampler()
outputs = samples[-2]
charset = pickle.load(open(config_dict['dict_path']))
new_charset = {}
for v in charset:
    new_charset[charset[v]] = v
charset = new_charset
print charset
for i in xrange(outputs.shape[1]):
    print "Sample number ", i, ": ",
    print ''.join(map(lambda x: charset[x], outputs[:,i]))

