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

def execute(cmd):
  ps = subprocess.Popen(cmd, shell=True,stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  output = ps.communicate()[0]
  return output

def plot(path_l, legend_l, output, color_l=['r','g','b','k'], \
        marker_l=['s', '+', 'o', 'v']):
  plot_l = list()
  for (pathi, path) in enumerate(path_l):
    time = list()
    acc = list()
    pl = list()
    try:
      for p in path:
        '''
        print p
        policy = PolicyResult(p)
        if count < 0:
          pl.append(policy)
          count += 1
        time.append(policy.ave_time())
        acc.append(policy.accuracy)
        '''
        p = p + '/policy.xml'
        print p
        acc.append(float(execute("cat %s | sed '0,/<accuracy>/d' | head -n 1" % p).split('\n')[0]))
        s =  execute("cat %s | tac | sed '0,/<\/time>/d' | head -n 1" % p)
        time.append(float(execute("cat %s | tac | sed '0,/<\/time>/d' | head -n 1" % p).split('\n')[0]))
      pair = sorted(zip(time, acc), key=lambda x: x[0])
      time, acc = zip(*pair)
      time, acc = (list(time), list(acc))
      p, = plt.plot(time, acc, '%s%s' % (color_l[pathi], marker_l[pathi]))
      plot_l.append(p)
      [time, acc] = zip(*sorted(zip(time,acc), key=lambda ta : ta[0]))
      print legend_l[pathi], time, acc
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
  scheme_l = ['gibbs', 'multi_policy']
  for scheme in scheme_l:
    target = '%s_%s'%(name, scheme)
    path = [path_in+'/test_policy/'+f for f in files if f.find(target) == 0 and f.find("_train") == -1 and f != target]
    path_l.append(path)
  plot(path_l, scheme_l, path_out+'/%s.png' % name)
  '''
  name_l = [[p.split('/')[-1] for p in path] for path in path_l]
  html = codecs.open(path_out+'/%s.html' % name, 'w', encoding='utf-8')
  html.write(PolicyResult.viscomp(list(itertools.chain(*policy_l)), \
                    list(itertools.chain(*name_l)), 'POS'))
  '''
