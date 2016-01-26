# -*- coding: UTF-8 -*-
import os
import pickle
import codecs

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
        files.append(conc.encode('utf-8'))
    files = "\n<*>LI<*>\n".join(files)
    g = open(out_path, 'wb')
    g.write(files)
    g.close()
            
prefix_path = '/pio/lscratch/1/i246059/language-model/data/plwiki_raw/plwiki/'


train_fnames = pickle.load(open('/pio/scratch/1/i246059/language-model/data/plwiki_raw/train_set_names', 'rb'))
test_fnames = pickle.load(open('/pio/scratch/1/i246059/language-model/data/plwiki_raw/test_set_names', 'rb'))
names = train_fnames + test_fnames
out_path = '/pio/scratch/1/i246059/language-model/data/plwiki_torch/input.txt'
build_set("TEST", names, prefix_path, out_path)
