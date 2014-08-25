import sys
f = file('letter_raw.data', 'r').readlines()
if len(sys.argv) >= 2:
  fold = int(sys.argv[1])
else:
  fold = 0
train = open('train%d' % fold, 'w')
test = open('test%d' % fold, 'w')
for line in f:
  line = line.replace('\n', '')  
  raw_line = line
  line = line.split('\t')
  this_fold = int(line[5])
  if this_fold == fold:
    stream = test
  else:
    stream = train
  stream.write(raw_line+'\n')
  if int(line[2]) == -1:
    stream.write('\n')

