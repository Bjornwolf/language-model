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

def build_set(name, fnames, prefix_path, out_path):
    print name
    for fname in fnames:
        full_path = prefix_path + fname
        with codecs.open(full_path, 'r') as f:
            conc = f.read().decode('utf-8')
        conc = filter(lambda x: x in acceptable_chars, conc)
        piece_no = 1
        i = int(fname[3:]) # bardzo brzydkie wykorzystanie formatu 'artXXXX'
        # robimy strukture prefix_path/nr_kawalka/i, bo i < 100
        while conc != '':
            try:
                os.mkdir(out_path + str(piece_no), 0755)
            except:
                pass
            with open(out_path + str(piece_no) + '/' + fname, 'wb+') as g:
               g.write(conc[:files_size].encode('utf-8'))
            conc = conc[files_size - overlap_size:]
            piece_no += 1
            
prefix_path = '/pio/lscratch/1/i246059/language-model/data/plwiki_raw/plwiki/'


train_fnames = pickle.load(open('/pio/lscratch/1/i246059/language-model/data/plwiki_raw/train_set_names', 'rb'))
out_path = '/pio/lscratch/1/i246059/language-model/data/plwiki_1gu/train/'
build_set("TRAIN", train_fnames, prefix_path, out_path)

test_fnames = pickle.load(open('/pio/lscratch/1/i246059/language-model/data/plwiki_raw/test_set_names', 'rb'))
out_path = '/pio/lscratch/1/i246059/language-model/data/plwiki_1gu/test/'
build_set("TEST", test_fnames, prefix_path, out_path)
