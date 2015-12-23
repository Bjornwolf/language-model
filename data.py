import os
import cPickle as pickle
from fuel.datasets import TextFile
from fuel.streams import DataStream
from fuel.transformers import Batch, Padding, Mapping
from fuel.schemes import ConstantScheme


def scan_files(path):
    files = []
    for dirname, _, filenames in os.walk(path):
        files += map(lambda x: dirname + '/' + x, filenames)
    print path, len(files)
    return files

def switch_first_two_axes(stream):
    result = []
    for array in stream:
        if array.ndim == 2:
            result.append(array.transpose(1, 0))
        else:
            result.append(array.transpose(1, 0, 2))
    return tuple(result)

def build_stream(files, alphabet, config):
    fuel_text = TextFile(files=files, dictionary=alphabet, 
                         bos_token=None, eos_token=None, 
                         unk_token=config['unknown_char'], level='character')
    stream = DataStream(fuel_text)
    if 'minibatches' in config:
        minibatches = config['minibatches']
        stream = Batch(stream, iteration_scheme=ConstantScheme(minibatches))
        stream = Mapping(Padding(stream), switch_first_two_axes)
    return stream
        

def build_datasets(config):
    alphabet = pickle.load(open(config['dict_path'], 'r'))
    train_files = scan_files(config['train_path'])
    train_stream = build_stream(train_files, alphabet, config)
    if 'valid_path' in config:
        valid_files = scan_files(config['valid_path'])
        valid_stream = build_stream(valid_files, alphabet, config)
    else:
        valid_stream = None
    return train_stream, valid_stream, alphabet
