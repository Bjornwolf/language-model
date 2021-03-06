from picklable_itertools import iter_, chain

from fuel.datasets import Dataset

import numpy
import pickle

import codecs

class BigFileIterator:
    def __init__(self, file_list):
        self.i = 0
        self.files = []
        for fname in file_list:
            handle = codecs.open(fname)
            content = handle.read().decode('utf-8')
            self.files += content.split('<*>LI<*>')
            handle.close()
        self.max_i = len(self.files)
    def __iter__(self):
        return self

    def next(self):
        if self.i == self.max_i:
            raise StopIteration
        self.i += 1
        return self.files[self.i-1]

class LineByLineIterator:
    def __init__(self, file_list):
        self.i = 0
        self.file_list = file_list
        self.handle = codecs.open(self.file_list[self.i])
        self.memory = []

    def __iter__(self):
        return self

    def remember(self, fragment):
        self.memory = fragment
    
    def forget(self):
        memo = self.memory
        self.memory = []
        return memo

    def next(self):
        sentence = self.handle.readline().decode('utf-8', 'ignore')
        while sentence == '\n':
            sentence = self.handle.readline().decode('utf-8', 'ignore') 
        if len(sentence) > 0:
            return sentence
        else:
            self.i += 1
            if self.i >= len(self.file_list):
                raise StopIteration
            self.handle = codecs.open(self.file_list[self.i])
            return self.handle.readline().decode('utf-8', 'ignore')

class TokenTextFile(Dataset):
    r"""
    Parameters
    ----------
    files : list of str
        1 sentence = 1 line
    dictionary : dict
    unk_token : str, optional
        Token to use when no appropriate token can be found in dictionary.
    """
    provides_sources = ('features',)
    example_iteration_scheme = None

    def __init__(self, files, dictionary, unk_token='<UNK>', 
                 batch_length=None, overlap=None):
        self.files = files
        self.dictionary = dictionary
        if unk_token not in dictionary:
            raise ValueError
        self.unk_token = unk_token
        self.batch_length = batch_length
        self.overlap = overlap
        if overlap is None:
            self.overlap = 0
        super(TokenTextFile, self).__init__()


    def open(self):
        return LineByLineIterator(self.files)

    def parse_sentence(self, sentence):
        data = []
        sentence_stack = ''
        for char in sentence.strip():
            sentence_stack += char
            if sentence_stack in self.dictionary:
                data.append(self.dictionary[sentence_stack])
                sentence_stack = ''
        return data

    def get_data(self, state=None, request=None):
        if request is not None:
            raise ValueError
        if self.batch_length is None:
            sentence = state.next()
            return (numpy.array(self.parse_sentence(sentence), dtype=numpy.int64),)
        else:
            initial_i = state.i
            data = state.forget()
            while len(data) < self.batch_length:
                sentence = state.next()
                if initial_i == state.i:
                    data += self.parse_sentence(sentence)
                else:
                    state.remember(self.parse_sentence(sentence))
                    return (numpy.array([data], dtype=numpy.int64),)
            state.remember(data[:self.batch_length - self.overlap])
            return (numpy.array(data[:self.batch_length], dtype=numpy.int64),)
        
class TextFile(Dataset):
    r"""Reads text files and numberizes them given a dictionary.

    Parameters
    ----------
    files : list of str
        The names of the files in order which they should be read. Each
        file is expected to have a sentence per line.
    dictionary : str or dict
        Either the path to a Pickled dictionary mapping tokens to integers,
        or the dictionary itself. At the very least this dictionary must
        map the unknown word-token to an integer.
    bos_token : str or None, optional
        The beginning-of-sentence (BOS) token in the dictionary that
        denotes the beginning of a sentence. Is ``<S>`` by default. If
        passed ``None`` no beginning of sentence markers will be added.
    eos_token : str or None, optional
        The end-of-sentence (EOS) token is ``</S>`` by default, see
        ``bos_taken``.
    unk_token : str, optional
        The token in the dictionary to fall back on when a token could not
        be found in the dictionary.
    level : 'word' or 'character', optional
        If 'word' the dictionary is expected to contain full words. The
        sentences in the text file will be split at the spaces, and each
        word replaced with its number as given by the dictionary, resulting
        in each example being a single list of numbers. If 'character' the
        dictionary is expected to contain single letters as keys. A single
        example will be a list of character numbers, starting with the
        first non-whitespace character and finishing with the last one.
    preprocess : function, optional
        A function which takes a sentence (string) as an input and returns
        a modified string. For example ``str.lower`` in order to lowercase
        the sentence before numberizing.

    """
    provides_sources = ('features',)
    example_iteration_scheme = None

    def __init__(self, files, dictionary, bos_token='<S>', eos_token='</S>',
                 unk_token='<UNK>', level='word', preprocess=None):
        self.files = files
        self.dictionary = dictionary
        if bos_token is not None and bos_token not in dictionary:
            raise ValueError
        self.bos_token = bos_token
        if eos_token is not None and eos_token not in dictionary:
            raise ValueError
        self.eos_token = eos_token
        if unk_token not in dictionary:
            raise ValueError
        self.unk_token = unk_token
        if level not in ('word', 'character'):
            raise ValueError
        self.level = level
        self.preprocess = preprocess
        super(TextFile, self).__init__()


    def open(self):
        return BigFileIterator(self.files)
        # return EagerFileIterator(self.files)
        # return LazyFileIterator(self.files)
        # return chain(*[iter_(open(f)) for f in self.files])

    def get_data(self, state=None, request=None):
        if request is not None:
            raise ValueError
        sentence = state.next()
        # sentence = state.next().read()
        # sentence = next(state)
        if self.preprocess is not None:
            sentence = self.preprocess(sentence)
        data = [self.dictionary[self.bos_token]] if self.bos_token else []
        if self.level == 'word':
            data.extend(self.dictionary.get(word,
                                            self.dictionary[self.unk_token])
                        for word in sentence.split())
        else:
            data.extend(self.dictionary.get(char,
                                            self.dictionary[self.unk_token])
                        for char in sentence.strip())
        if self.eos_token:
            data.append(self.dictionary[self.eos_token])
        return (data,)
