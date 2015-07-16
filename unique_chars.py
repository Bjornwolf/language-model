f = open('biblia.txt', 'r')
letters = set('')
line = 'abc'
while line != '':
    line = f.readline()
    sl = set(line)
    letters = letters | sl
print letters


