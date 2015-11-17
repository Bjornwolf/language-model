import os
import pickle
print "TRAIN"
train_fnames = pickle.load(open('../data/plwiki_raw/train_set_names', 'rb'))
prefix_path = '../data/plwiki_raw/plwiki/'
out_path = '../data/plwiki_1g/train/'
for fname in train_fnames:
   full_path = prefix_path + fname
   f = open(full_path, 'r')
   conc = reduce(lambda a, b: a + b, f.readlines())
   f.close()
   cf = 1
   i = int(fname[3:])
   while conc != '':
      try:
         os.mkdir(out_path + str(cf), 0755)
      except:
         pass
      try:
         os.mkdir(out_path + str(cf) + '/' + str(i//1000), 0755)
      except:
         pass
      g = open(out_path + str(cf) + '/' + str(i//1000) + '/' + fname, 'w+')
      g.write(conc[:1000])
      conc = conc[500:]
      cf += 1
      g.close()
print "TEST"
test_fnames = pickle.load(open('../data/plwiki_raw/test_set_names', 'rb'))
prefix_path = '../data/plwiki_raw/plwiki/'
out_path = '../data/plwiki_1g/test/'
for fname in test_fnames:
   print fname
   full_path = prefix_path + fname
   f = open(full_path, 'r')
   conc = reduce(lambda a, b: a + b, f.readlines())
   f.close()
   cf = 1
   i = int(fname[3:])
   while conc != '':
      try:
         os.mkdir(out_path + str(cf), 0755)
      except:
         pass
      try:
         os.mkdir(out_path + str(cf) + '/' + str(i//1000), 0755)
      except:
         pass
      g = open(out_path + str(cf) + '/' + str(i//1000) + '/' + fname, 'w+')
      g.write(conc[:1000])
      conc = conc[500:]
      cf += 1
      g.close()
