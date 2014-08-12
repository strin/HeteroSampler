import os, sys

def read_tagcounts(filename, mode='POS'):
  f = file(filename).readlines()
  tagcount = dict()
  for line in f:
    tokens = line.replace('\n', '').split(' ')
    if len(tokens) < 2:
      continue
    if mode == 'POS':
      [word, pos] = tokens[:2]
      tag = pos
    else:
      [word, pos, pos2, ner] = tokens[:4]
      tag = ner
    if not tagcount.has_key(word):
      tagcount[word] = dict()
    if not tagcount[word].has_key(tag):
      tagcount[word][tag] = 0
    tagcount[word][tag] += 1
  return tagcount

def read_tagposterior(filename, mode='POS'):
  tagcount = read_tagcounts(filename, mode)
  for key in tagcount.keys():
    vsum = sum(tagcount[key].values())
    for tagkey in tagcount[key].keys():
      tagcount[key][tagkey] /= float(vsum)
      tagcount[key][tagkey] = float('%0.4f' % tagcount[key][tagkey])
  return tagcount

if __name__ == '__main__':
  filename = sys.argv[1]
  tagcount = read_tagposterior(filename, 'NER')
  print tagcount

