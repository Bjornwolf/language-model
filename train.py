import sys

from config import read_config
from data import build_datasets
from model import build_model, build_algorithm
from monitor import build_extensions


config_dict = read_config(sys.argv[1])

if config_dict['working_mode'] == 'train_new':
    train, valid, alphabet = build_datasets(config_dict)
    cost = build_model(len(alphabet), config_dict)
    algorithm = build_algorithm(cost, config_dict)
    extensions = build_extensions(cost, algorithm, config_dict)
    main_loop = MainLoop(algorithm=algorithm, data_stream=train,
                         model=Model(cost), extensions=extensions)
    main_loop.run()

elif config_dict['working_mode'] == 'train_resume':
    # TODO
