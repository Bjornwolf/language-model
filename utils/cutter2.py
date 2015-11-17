import os
import pickle
print "TRAIN"
train_fnames = ['lagowski']
prefix_path = '../data/bl/'
out_path = '../data/bl/'
for fname in train_fnames:
   full_path = prefix_path + fname
   f = open(full_path, 'r')
   conc = reduce(lambda a, b: a + b, f.readlines())
   f.close()
   cf = 1
   while conc != '':
      g = open(out_path + str(cf), 'w+')
      g.write(conc[:1000])
      conc = conc[500:]
      cf += 1
      g.close()
   print cf
