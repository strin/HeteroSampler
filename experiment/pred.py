import xml.etree.ElementTree as ElementTree
import numpy as np
import os, sys
import numpy.random as npr
import matplotlib.pyplot as plt

tree = ElementTree.parse('stopdata.xml')
root = tree.getroot()

feat = list()   # feature is a list of dict.
RNow  = list()   # current reward (list of float)
RFuture = list()  # expected reward (list of float)

for data in root:
  f = dict()
  for item in data:
    if item.tag == "RNow":
      RNow.append(float(item.text))
    elif item.tag == "RFuture":
      RFuture.append(float(item.text))
    elif item.tag == "entry":
      # if item.attrib['name'] in ['len-prob-tag-bigram', 'max-freq', 'ave-freq', \
      #                     'max-ent', 'ave-ent', 'len-inv-stopornot']:
      #                            'len-stopornot', 'bias-stopornot']:
      # if item.attrib['name'] in ['len-stopornot']:
        f[item.attrib['name']] = float(item.attrib['value'])
  feat.append(f)
value = [now - future for (now, future) in zip(RNow, RFuture)]

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

normalize_feat(feat)
for it in range(num_pass):
  eff_acc = 0.0
  grad = dict()
  for ni in npr.permutation(range(train_size)):
    f = feat[ni]
    r_now = RNow[ni]
    r_future = RFuture[ni]
    v = r_now - r_future
    # print 'f = ', f
    # print 'v = ', v
    # print 'r_now', r_now, 'r_future', r_future
    resp = sigmoid(score(param, f))

    # eff_acc += resp * (v >= 0) + (1 - resp) * (v < 0)         # use effective accuracy.
    eff_acc += r_now * resp + r_future * (1-resp)   # use effective reward.

    # learning strategy 1. using real response to train.
    updateMap(grad, f, (r_now - r_future) * resp * (1 - resp))
    # learning strategy 2. using binary response to train.
    # updateMap(grad, f, ((v >= 0) * (1 - resp) - (v < 0) * resp)/train_size);

    # learning strategy 3. minimize expected error.
    # updateMap(grad, f, ((v >= 0) - (v < 0)) * resp * (1 - resp));

  # apply grad using AdaGrad.
  for key in grad:
    updateKey(G2, key, grad[key] * grad[key])
    updateKey(param, key, eta * grad[key] / np.sqrt(1e-4 + G2[key])); # use adagrad.
    # updateKey(param, key, 1e-4 * grad[key])   # use grad.

  eff_acc /= train_size

  acc = 0.0
  truth = 0.0;
  reward_all_stop = 0.0
  reward_all_go = 0.0
  for ni in range(train_size+1, data_size):
    f = feat[ni]
    r_now = RNow[ni]
    r_future = RFuture[ni]
    v = r_now - r_future
    if v >= 0:
      truth += r_now
    else:
      truth += r_future
    reward_all_stop += r_now
    reward_all_go += r_future
    resp = sigmoid(score(param, f))
    acc += r_now * resp + r_future * (1 - resp)                               # use reward.

  acc /= (data_size-train_size)  
  truth /= (data_size-train_size)
  reward_all_go /= (data_size-train_size)
  reward_all_stop /= (data_size-train_size)

  print 'iter', it, 'eff acc', eff_acc, 'test acc', acc, 'truth', truth, 'all_stop', \
            reward_all_stop, 'all_go', reward_all_go
  acc_list.append(acc)
  eff_acc_list.append(eff_acc)

print param
plt.plot(range(num_pass), acc_list, 'r-')
plt.plot(range(num_pass), eff_acc_list, 'g-')
plt.title('train-test accuracy')
plt.show()
      




  
