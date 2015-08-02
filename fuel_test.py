from fuel.datasets import Dataset, TextFile

def get_unique_chars(filelist):
    letters = set('')
    for path in filelist:
        f = open(path, 'r')
        line = '^'
        while line != '':
            line = f.readline()
            sl = set(line)
            letters = letters | sl
    letters = list(letters)
    result = {}
    for i in range(len(letters)):
        result[letters[i]] = i
    return result
        
art_count = 359586

files = ["data/plwiki/art" + str(i) for i in range(1, art_count + 1)]
data = TextFile(files = files,
                dictionary = get_unique_chars(files),
                bos_token = None,
                eos_token = None,
                unk_token = "^",
                level = 'character')
print "OK"
