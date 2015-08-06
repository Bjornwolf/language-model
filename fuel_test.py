from fuel.datasets import Dataset, TextFile
from fuel.streams import DataStream
#import dill as pickle
import pickle

def get_unique_chars(filelist):
    letters = set('')
    for path in filelist:
        f = open(path, 'r')
        line = '^'
        while line != '':
            line = f.readline()
            sl = set(line)
            letters = letters | sl
        f.close()
    letters = list(letters)
    result = {'<UNK>': 0}
    for (i, letter) in enumerate(letters):
        result[letter] = i + 1
    return result
        
art_count = 100

files = ["data/plwiki/art" + str(i) for i in range(1, art_count + 1)]
data = TextFile(files = files,
                dictionary = get_unique_chars(files),
                bos_token = None,
                eos_token = None,
                unk_token = '<UNK>',
                level = 'character')

print "OK"
cnt = 0

for one in DataStream(data).get_epoch_iterator():
    cnt += 1
    print one
print cnt
