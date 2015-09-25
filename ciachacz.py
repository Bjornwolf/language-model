import os
i = 1
while True:
   if i % 10000 == 0:
      print i
   f = open('art' + str(i), 'r')
   conc = reduce(lambda a, b: a + b, f.readlines())
   cf = 1
   while conc != '':
      try:
         os.mkdir(str(cf), 0755)
      except:
         pass
      try:
         os.mkdir(str(cf) + '/' + str(i//1000), 0755)
      except:
         pass
      g = open(str(cf) + '/' + str(i//1000) + '/art' + str(i), 'wb')
      g.write(conc[:1000])
      conc = conc[500:]
      cf += 1
      g.close()
   i += 1
   f.close()

