import os
import pickle
import codecs.open as open

files_size = 300
overlap_size = 100

def build_set(name, fnames, prefix_path, out_path):
    print name
    for fname in fnames:
        full_path = prefix_path + fname
        with open(full_path, 'r') as f:
            conc = f.read()
        piece_no = 1
        i = int(fname[3:]) # bardzo brzydkie wykorzystanie formatu 'artXXXX'
        # robimy strukture prefix_path/nr_kawalka/i, bo i < 100
        while conc != '':
            try:
                os.mkdir(out_path + str(piece_no), 0755)
            except:
                pass
            with open(out_path + str(piece_no) + '/' + fname, 'w+') as g:
                g.write(conc[:files_size])
            conc = conc[files_size - overlap_size:]
            piece_no += 1
            
prefix_path = '../data/plwiki_raw/plwiki/'


train_fnames = pickle.load(open('../data/plwiki_raw/train_set_names', 'rb'))
out_path = '../data/plwiki_1gu/train/'
build_set("TRAIN", train_fnames, prefix_path, out_path)

test_fnames = pickle.load(open('../data/plwiki_raw/test_set_names', 'rb'))
out_path = '../data/plwiki_1gu/test/'
build_set("TEST", test_fnames, prefix_path, out_path)
