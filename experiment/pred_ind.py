import xml.etree.ElementTree as ElementTree
import numpy as np
import os, sys
import numpy.random as npr
import matplotlib.pyplot as plt

tree = ElementTree.parse('stopdata_ind.xml')
root = tree.getroot()

feat = list()   # feature is a list of dict.
RNow  = list()   # current reward (list of float)
RFuture = list()  # expected reward (list of float)

for data in root:
  f = dict()
  for item in data:
    if item.tag == "RNow":
      RNow.append(float(item.text))
    elif item.tag == "feat":
      for entry in item:
        f[entry.attrib['name']] = float(entry.attrib['value'])
  feat.append(f)
value = RNow

def getMap(dic, key):
  if not dic.has_key(key):
    return 0.0
  return dic[key]

def updateMap(dict_to, dict_from, factor):
  for key in dict_from:
    updateKey(dict_to, key, dict_from[key] * factor)

def updateKey(dict_to, key, value):
  if not dict_to.has_key(key):
    dict_to[key] = 0.0
  dict_to[key] += value


def score(param, feat):
  ret = 0.0
  for key in feat:
    if param.has_key(key):
      ret += param[key] * feat[key]
  return ret

def sigmoid(z):
  return 1./(1+np.exp(-z))

def plot_corr(key):
  x = [item[key] for item in feat]
  y = value
  plt.clf()
  plt.plot(x, y, 'r.')
  plt.xlabel('feat')
  plt.ylabel('val')
  plt.title(key)
  plt.savefig('plot_%s.pdf' % key)
  os.system('open plot_%s.pdf' % key)
  
def normalize_feat(feat):
  for key in feat[0]:
    if key == "bias-stopornot":
      continue
    fv = np.array([f[key] for f in feat])
    mean = np.mean(fv)
    for f in feat:
      f[key] -= mean
    std = np.std(fv-mean)
    print 'key', key, std
    if std == 0:
      continue
    for f in feat:
      f[key] = f[key] / std * 10

def get_featvec(feat):
  ret = list()
  keys = feat[0].keys()
  for f in feat:
    row = list()
    for key in keys:
      row.append(f[key])
    ret.append(row)
  return ret

      
trainp = 0.8
num_pass = 300
eta = 1

data_size = len(feat)
train_size = int(data_size * trainp)




param = dict()
G2 = dict()

acc_list = list()
eff_acc_list = list()

def plot_all_corr(name):
  if name == "":
    for key in feat[0]:
      print key
      plot_corr(key)
  else:
    plot_corr(name)
  exit(0)

# plot_all_corr("len-stopornot")
plot_all_corr("")

