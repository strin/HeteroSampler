import os, sys

def read_tagcounts(filename, mode="POS"):
  f = file(filename).readlines()
  tagcount = dict()
  for line in f:
    tokens = line.replace('\n', '').split(' ')
    if len(tokens) < 4:
      continue
    [word, pos, pos2, ner] = tokens[:4]
    if not tagcount.has_key(word):
      tagcount[word] = dict()
    if mode == "NER":
      tag = ner
    else:
      tag = pos
    if not tagcount[word].has_key(tag):
      tagcount[word][tag] = 0
    tagcount[word][tag] += 1
  return tagcount

def read_tagposterior(filename, mode="POS"):
  tagcount = read_tagcounts(filename)
  for key in tagcount.keys():
    vsum = sum(tagcount[key].values())
    for tagkey in tagcount[key].keys():
      tagcount[key][tagkey] /= float(vsum)
      tagcount[key][tagkey] = float('%0.4f' % tagcount[key][tagkey])
  return tagcount

if __name__ == '__main__':
  filename = sys.argv[1]
  # tagcount = read_tagcounts(filename, 'NER')
  tagcount = read_tagposterior(filename, 'NER')
  print tagcount

