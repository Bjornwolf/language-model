# -*- coding: UTF-8 -*-
import sys
import yaml
import numpy as np
from data import build_datasets
from model import build_model, build_algorithm
from monitor import build_extensions
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.extensions.saveload import Load
import cPickle as pickle

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
print dir(main_loop)
print dir(main_loop.log)
#print main_loop.log.keys()
print main_loop.log[1]
t_xs = main_loop.log.keys()[1:]
v_xs = filter(lambda x: 'valid_bits_per_character' in main_loop.log[x], t_xs)
bpc = map(lambda x: main_loop.log[x]['bits_per_character'], t_xs)
gn = map(lambda x: main_loop.log[x]['total_gradient_norm'], t_xs)
lr = map(lambda x: main_loop.log[x]['learning_rate'], t_xs)
#print [np.percentile(gn, i) for i in range(101)]
vbpc = map(lambda x: main_loop.log[x]['valid_bits_per_character'], v_xs)
print vbpc[-1]
blob = ((t_xs, bpc), (v_xs, vbpc), (t_xs, gn), (t_xs, lr), 1.75)
pickle.dump(blob, open('plots/' + sys.argv[1][:-5], 'w'))
