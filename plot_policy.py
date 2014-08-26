import numpy as np
import os, sys
import numpy.random as npr
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools
import codecs
import subprocess
mpl.use('Agg')
from stat_policy import *

def plot(path_l, legend_l, output, color_l=['r','g','b','k'], \
        marker_l=['s', '+']):
  plot_l = list()
  policy_l = list()
  for (pathi, path) in enumerate(path_l):
    time = list()
    acc = list()
    pl = list()
    try:
      count = 0
      for p in path:
        print p
        policy = PolicyResult(p)
        if count < 3:
          pl.append(policy)
          count += 1
        time.append(policy.ave_time())
        acc.append(policy.accuracy)
      policy_l.append(pl)
      pair = sorted(zip(time, acc), key=lambda x: x[0])
      time, acc = zip(*pair)
      time, acc = (list(time), list(acc))
      p, = plt.plot(time, acc, '%s%s' % (color_l[pathi], marker_l[pathi]))
      plot_l.append(p)
      [time, acc] = zip(*sorted(zip(time,acc), key=lambda ta : ta[0]))
      print time, acc
      (time, acc) = (list(time), list(acc))
      plt.plot(time, acc, '%s-' % (color_l[pathi]))
      plt.plot(time, acc, '%s%s' % (color_l[pathi], marker_l[pathi]))
    except Exception as e:
      print e
      continue
  plt.xlabel('# Gibbs update')
  plt.ylabel('Score')
  plt.legend(plot_l, legend_l, loc=4)
  plt.savefig(output)
  return policy_l
  
if __name__ == '__main__':
  name = sys.argv[1]
  if len(sys.argv) >= 3:
    path_in = sys.argv[2]
  else:
    path_in = '.'
  if len(sys.argv) >= 4:
    path_out = sys.argv[3]
  else:
    path_out = '.'
  files = os.listdir(path_in+'/test_policy/')
  files = [f[0] for f in sorted([(f, os.stat(path_in+'/test_policy/'+f)) for f in files], key=lambda x: x[1].st_ctime)]
  path_l = list()
  scheme_l = ['gibbs', 'multi_policy', 'oracle']
  for scheme in scheme_l:
    path = [path_in+'/test_policy/'+f for f in files if f.find('%s_%s'%(name, scheme)) == 0]
    path_l.append(path)
  policy_l = plot(path_l, scheme_l, path_out+'/%s.png' % name)
  name_l = [[p.split('/')[-1] for p in path] for path in path_l]
  html = codecs.open(path_out+'/%s.html' % name, 'w', encoding='utf-8')
  html.write(PolicyResult.viscomp(list(itertools.chain(*policy_l)), \
                    list(itertools.chain(*name_l)), 'POS'))
