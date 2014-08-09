import numpy as np
import os, sys
import numpy.random as npr
import matplotlib.pyplot as plt
from stat_stop import *
from stat_policy import *

thres_l = [0.5, 1, 1.5, 2, 2.5, 3, 1000]
c_l = [0.0, 0.01, 0.1, 0.20, 0.3, 1]
T_l = [1, 2, 3]

if len(sys.argv) < 2:
  print 'toy', 'toy0', 'plot_toy'
  exit(0)

if sys.argv[1] == "toy_entropy":
    for thres in thres_l:
      cmd = '''./policy --inference Gibbs --policy entropy --name test_policy/entropy_%0.2f \
      --testCount 100 --threshold %0.2f --numThreads 10''' % (thres, thres)
      print cmd
      os.system(cmd)  
      policy = PolicyResult('test_policy/entropy_%0.2f' % thres)
      print 'time: ', policy.ave_time(), 'acc: ', policy.accuracy
if sys.argv[1] == "toy_cyclic":
  for c in c_l:
      cmd = '''./policy --inference Gibbs --policy cyclic --name test_policy/cyclic_%f \
      --trainCount 100 --testCount 100 --c %f --numThreads 10 --eta 1 --K 5''' % (c, c)
      print cmd
      os.system(cmd)  
      policy = PolicyResult('test_policy/cyclic_%f' % c)
      print 'time: ', policy.ave_time(), 'acc: ', policy.accuracy
elif sys.argv[1] == "toy_gibbs":
  for T in T_l:
    cmd = '''./policy --inference Gibbs --policy gibbs --name test_policy/gibbs_T%d --T %d --testCount 100 --numThreads 4''' % (T, T)
    print cmd
    os.system(cmd)  
    policy = PolicyResult('test_policy/gibbs_T%d' % T)
    print 'time: ', policy.ave_time(), 'acc: ', policy.accuracy
elif sys.argv[1] == 'plot_toy':
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
elif sys.argv[1] == "toy0":
  for T in T_l:
    cmd = '''./stop --inference Gibbs --T %d --name gibbs0_T%d --numThreads 10 \
    --trainCount 1000 --testCount 1000 --adaptive false''' % (T, T)
    print cmd
    os.system(cmd)  
    stop = StopResult('gibbs0_T%d' % T)
    print 'T = ', T, 'acc = ', stop.accuracy
