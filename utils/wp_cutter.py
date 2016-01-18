# -*- coding: UTF-8 -*-
import os
import pickle
import codecs
import random

files_size = 300
overlap_size = 100
acceptable_chars = u'abcdefghijklmnopqrstuvwxyz'
acceptable_chars += u'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
acceptable_chars += u'1234567890`~-_=+[]{}\|;:,<.>/?\n"'
acceptable_chars += u"'!@#$%^&*()"
acceptable_chars += u'ęóąśłżźńć'
acceptable_chars += u'ĘÓĄŚŁŻŹŃĆ \t'

def build_set(fnames, train_path, valid_path, p):
    train_files = []
    valid_files = []
    for fname in fnames:
        with codecs.open(fname, 'r') as f:
            conc = f.read().decode('utf-8')
        conc = filter(lambda x: x in acceptable_chars, conc)
        while conc != '':
            s = random.random()
            if s < p:
                valid_files.append(conc[:files_size].encode('utf-8'))
            else:
                train_files.append(conc[:files_size].encode('utf-8'))
            conc = conc[files_size - overlap_size:]
    train_files = "<*>LI<*>".join(train_files)
    g = open(train_path, 'wb')
    g.write(train_files)
    g.close()
    valid_files = "<*>LI<*>".join(valid_files)
    g = open(valid_path, 'wb')
    g.write(valid_files)
    g.close()
                


fnames = ['/pio/scratch/1/i246059/language-model/data/war_peace']
tr_path = '/pio/scratch/1/i246059/language-model/data/wp/train/data'
vd_path = '/pio/scratch/1/i246059/language-model/data/wp/test/data'
build_set(fnames, tr_path, vd_path, 0.04)

