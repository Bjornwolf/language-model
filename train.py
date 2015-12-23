import sys
import yaml
from config import read_config
from data import build_datasets
from model import build_model, build_algorithm
from monitor import build_extensions


config_dict = yaml.load(open(sys.argv[1], 'r'))
print config_dict

if config_dict['working_mode'] == 'train_new':
    train, valid, alphabet = build_datasets(config_dict)
    generator, cost = build_model(len(alphabet), config_dict)
    algorithm = build_algorithm(generator, cost, config_dict)
    extensions = build_extensions(cost, algorithm, config_dict)
    main_loop = MainLoop(algorithm=algorithm, data_stream=train,
                         model=Model(cost), extensions=extensions)
    main_loop.run()

elif config_dict['working_mode'] == 'train_resume':
    # TODO
