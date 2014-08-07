import numpy as np
import os, sys
import numpy.random as npr
import matplotlib.pyplot as plt
from stat_stop import *

c_l = [0.001, 0.01, 0.03, 0.07, 0.1, 0.3, 1]
T_l = [0, 1, 2, 3, 4, 5]

if len(sys.argv[1]) < 2:
  print 'toy', 'toy0', 'plot_toy'

if sys.argv[1] == "toy":
  for c in c_l:
    cmd = '''./stop --inference Gibbs --T 5 --K 3 --B 0 --eta 0.01 \
 --c %f --name gibbs_c%f --numThreads 3 --trainCount 100 \
--testCount 100''' % (c, c)
    print cmd
    os.system(cmd)  
    stop = StopResult('gibbs_c%f' % c)
    print 'c = ', c, 'acc = ', stop.accuracy
elif sys.argv[1] == "toy0":
  for T in T_l:
    cmd = '''./stop --inference Gibbs --T %d --name gibbs0_T%d --numThreads 3 \
    --trainCount 100 --testCount 100 --adaptive false''' % (T, T)
    print cmd
    os.system(cmd)  
    stop = StopResult('gibbs0_T%d' % T)
    print 'T = ', T, 'acc = ', stop.accuracy
elif sys.argv[1] == 'plot_toy':
    time = list()
    acc = list()
    for c in c_l:
      stop = StopResult('gibbs_c%f' % c)
      time.append(stop.ave_time())
      acc.append(stop.accuracy)
    p1, = plt.plot(time, acc, 'r-') 
    time = list()
    acc = list()
    for T in T_l:
      stop = StopResult('gibbs0_T%d' % T)
      time.append(stop.ave_time())
      acc.append(stop.accuracy)
    p2, = plt.plot(time, acc, 'b-')
    plt.legend([p1, p2], ['Stop', 'Baseline'])
    plt.show()
