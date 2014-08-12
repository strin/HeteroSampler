import numpy as np
import os, sys
import numpy.random as npr
from stat_policy import *

thres_l = [0.5, 1, 1.5, 2, 2.5, 3, 1000]
c_l = [0.0, 0.01, 0.1, 0.20, 0.3, 1]
T_l = [1, 2, 3, 4]

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
if sys.argv[1] == "wsj_entropy":
    for thres in thres_l:
      cmd = '''./policy --inference Gibbs --policy entropy \
      --name test_policy/wsj_entropy_%0.2f --train data/wsj/wsj-pos.train \
      --test data/wsj/wsj-pos.test --threshold %0.2f --numThreads 10 \
      --model model/wsj_gibbs.model ''' % (thres, thres)
      print cmd
      os.system(cmd)  
      policy = PolicyResult('test_policy/wsj_entropy_%0.2f' % thres)
      print 'time: ', policy.ave_time(), 'acc: ', policy.accuracy
if sys.argv[1] == "wsj_cyclic":
  for c in c_l:
      cmd = '''./policy --inference Gibbs --policy cyclic \
      --name test_policy/wsj_cyclic_%f --train data/wsj/wsj-pos.train \
      --test data/wsj/wsj-pos.test --c %f --numThreads 10 --eta 1 --K 10 \
      --model model/wsj_gibbs.model''' % (c, c)
      print cmd
      os.system(cmd)  
      policy = PolicyResult('test_policy/wsj_cyclic_%f' % c)
      print 'time: ', policy.ave_time(), 'acc: ', policy.accuracy
elif sys.argv[1] == "wsj_gibbs":
  for T in T_l:
    cmd = '''./policy --inference Gibbs --policy gibbs --name test_policy/wsj_gibbs_T%d \
    --T %d --numThreads 10 --train data/wsj/wsj-pos.train \
    --test data/wsj/wsj-pos.test --model model/wsj_gibbs.model ''' % (T, T)
    print cmd
    os.system(cmd)  
    policy = PolicyResult('test_policy/wsj_gibbs_T%d' % T)
    print 'time: ', policy.ave_time(), 'acc: ', policy.accuracy
if sys.argv[1] == "ner_entropy":
    for thres in thres_l:
      cmd = '''./policy --inference Gibbs --windowL 2 --policy entropy \
      --name test_policy/ner_entropy_%0.2f --threshold %0.2f --numThreads 10 \
      --model model/ner_gibbs.model --mode NER''' % (thres, thres)
      print cmd
      os.system(cmd)  
      policy = PolicyResult('test_policy/ner_entropy_%0.2f' % thres)
      print 'time: ', policy.ave_time(), 'acc: ', policy.accuracy
if sys.argv[1] == "ner_cyclic":
  for c in c_l:
      cmd = '''./policy --inference Gibbs --policy cyclic \
      --name test_policy/ner_cyclic_%f --c %f --numThreads 10 --eta 1 --K 10 \
      --model model/ner_gibbs.model --windowL 2 --mode NER ''' % (c, c)
      print cmd
      os.system(cmd)  
      policy = PolicyResult('test_policy/ner_cyclic_%f' % c)
      print 'time: ', policy.ave_time(), 'acc: ', policy.accuracy
if sys.argv[1] == "ner_cyclic_value":
  for c in c_l:
      cmd = '''./policy --inference Gibbs --policy cyclic_value \
      --name test_policy/ner_cyclic_value_%f --c %f --numThreads 10 --eta 1 --K 10 \
      --model model/ner_gibbs.model --windowL 2 --mode NER ''' % (c, c)
      print cmd
      os.system(cmd)  
      policy = PolicyResult('test_policy/ner_cyclic_value_%f' % c)
      print 'time: ', policy.ave_time(), 'acc: ', policy.accuracy
elif sys.argv[1] == "ner_gibbs":
  for T in T_l:
    cmd = '''./policy --inference Gibbs --policy gibbs --name test_policy/ner_gibbs_T%d \
    --T %d --numThreads 10 --model model/ner_gibbs.model --mode NER --windowL 2''' % (T, T)
    print cmd
    os.system(cmd)  
    policy = PolicyResult('test_policy/ner_gibbs_T%d' % T)
    print 'time: ', policy.ave_time(), 'acc: ', policy.accuracy
elif sys.argv[1] == "toy0":
  for T in T_l:
    cmd = '''./stop --inference Gibbs --T %d --name gibbs0_T%d --numThreads 10 \
    --trainCount 1000 --testCount 1000 --adaptive false''' % (T, T)
    print cmd
    os.system(cmd)  
    stop = StopResult('gibbs0_T%d' % T)
    print 'T = ', T, 'acc = ', stop.accuracy
