import numpy as np
import os, sys
import numpy.random as npr
import matplotlib.pyplot as plt
from stat_stop import *
from stat_policy import *

if sys.argv[1] == 'plot_toy':
  if len(sys.argv) > 2:
    path = sys.argv[2]
  else:
    path = '.'
  time = list()
  acc = list()
  for thres in thres_l:
    policy = PolicyResult(path+'/test_policy/entropy_%0.2f' % thres)
    time.append(policy.ave_time())
    acc.append(policy.accuracy)
  p1, = plt.plot(time, acc, 'rx') 
  time = list()
  acc = list()
  for T in T_l:
    stop = PolicyResult(path+'/test_policy/gibbs_T%d' % T)
    time.append(stop.ave_time())
    acc.append(stop.accuracy)
  p2, = plt.plot(time, acc, 'b-')
  time = list()
  acc = list()
  for c in c_l:
    stop = PolicyResult(path+'/test_policy/cyclic_%f' % c)
    time.append(stop.ave_time())
    acc.append(stop.accuracy)
  p3, = plt.plot(time, acc, 'go')
  plt.legend([p1, p2, p3], ['Threshold', 'Baseline', 'Policy'])
  plt.show()
