# -*- coding: UTF-8 -*-
import os
import pickle
import codecs

files_size = 300
overlap_size = 100
acceptable_chars = u'abcdefghijklmnopqrstuvwxyz'
acceptable_chars += u'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
acceptable_chars += u'1234567890`~-_=+[]{}\|;:,<.>/?\n"'
acceptable_chars += u"'!@#$%^&*()"
acceptable_chars += u'ęóąśłżźńć'
acceptable_chars += u'ĘÓĄŚŁŻŹŃĆ \t'

def build_set(name, fnames, prefix_path, out_path):
    print name
    files = []
    for fname in fnames:
        full_path = prefix_path + fname
        with codecs.open(full_path, 'r') as f:
            conc = f.read().decode('utf-8')
        conc = filter(lambda x: x in acceptable_chars, conc)
        while conc != '':
            files.append(conc[:files_size].encode('utf-8'))
            conc = conc[files_size - overlap_size:]
    files = "<*>LI<*>".join(files)
    g = open(out_path, 'wb')
    g.write(files)
    g.close()
            
prefix_path = ''


train_fnames = ['/pio/scratch/1/i246059/language-model/data/bl_train']
out_path = '/pio/scratch/1/i246059/language-model/data/blag/train/data'
build_set("TRAIN", train_fnames, prefix_path, out_path)

test_fnames = ['/pio/scratch/1/i246059/language-model/data/bl_test']
out_path = '/pio/scratch/1/i246059/language-model/data/blag/test/data'
build_set("TEST", test_fnames, prefix_path, out_path)
