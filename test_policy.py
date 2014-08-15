import numpy as np
import os, sys
import numpy.random as npr
from farm import *
from stat_policy import *

thres_l = [0.5, 1, 1.5, 2, 2.5, 3, 1000]
T_l = [1, 2, 3, 4]

def pos_ner_gibbs(w, test_count, T):
    cmd = '''./policy --inference Gibbs --policy gibbs --name test_policy/pos_ner_w%d_tc%d_gibbs_T%d \
    --T %d --numThreads 5 --model model/ner_pos_gibbs_w%d.model --scoring Acc --windowL %d --testCount %d \
    --verbose false --train data/eng_pos_ner/train --test data/eng_pos_ner/test''' \
    % (w, test_count, T, T, w,  w, test_count)
    print cmd
    os.system(cmd)  
    policy = PolicyResult('test_policy/pos_ner_w%d_tc%d_gibbs_T%d' % (w, test_count, T))
    print 'time: ', policy.ave_time(), 'acc: ', policy.accuracy

def pos_ner_policy(w, count, c):
    cmd = '''./policy --inference Gibbs --policy cyclic_value --name test_policy/pos_ner_w%d_tc%d_policy_c%f \
    --c %f --numThreads 5 --model model/ner_pos_gibbs_w%d.model --scoring Acc --windowL %d --testCount %d \
    --trainCount %d --verbose false --train data/eng_pos_ner/train --test data/eng_pos_ner/test''' \
    % (w, count, c, c, w,  w, count, count)
    print cmd
    os.system(cmd)  
    policy = PolicyResult('test_policy/pos_ner_w%d_tc%d_policy_c%f' % (w, count, c))
    print 'time: ', policy.ave_time(), 'acc: ', policy.accuracy

def ner_gibbs(w, f, test_count, T):
    cmd = '''./policy --inference Gibbs --policy gibbs --name test_policy/ner_w%d_f%d_tc%d_gibbs_T%d \
    --T %d --numThreads 5 --model model/ner_gibbs_w%d_d2_f%d.model --scoring NER --windowL %d --testCount %d \
    --depthL 2 --factorL %d --verbose false --train data/eng_ner/train --test data/eng_ner/test''' \
    % (w, f, test_count, T, T, w, f,  w, test_count, f)
    print cmd
    os.system(cmd)  
    policy = PolicyResult('test_policy/ner_w%d_f%d_tc%d_gibbs_T%d' % (w, f, test_count, T))
    print 'time: ', policy.ave_time(), 'acc: ', policy.accuracy

def ner_policy(w, f, count, c):
    cmd = '''./policy --inference Gibbs --policy cyclic_value --name test_policy/ner_w%d_f%d_tc%d_policy_c%f \
    --c %f --numThreads 5 --model model/ner_gibbs_w%d_d2_f%d.model --scoring NER --windowL %d --testCount %d \
    --depthL 2 --factorL %d --trainCount %d --verbose false --train data/eng_ner/train --test data/eng_ner/test''' \
    % (w, f, count, c, c, w, f,  w, count, f,  count)
    print cmd
    os.system(cmd)  
    policy = PolicyResult('test_policy/ner_w%d_f%d_tc%d_policy_c%f' % (w, f, count, c))
    print 'time: ', policy.ave_time(), 'acc: ', policy.accuracy

def czech_gibbs(w, test_count, T):
    cmd = '''./policy --inference Gibbs --policy gibbs --name test_policy/czech_w%d_tc%d_gibbs_T%d \
    --T %d --numThreads 5 --model model/czech_gibbs_w%d.model --scoring Acc --windowL %d --testCount %d \
    --verbose false --train data/czech_ner/train --test data/czech_ner/test''' \
    % (w, test_count, T, T, w,  w, test_count)
    print cmd
    os.system(cmd)  
    policy = PolicyResult('test_policy/czech_w%d_tc%d_gibbs_T%d' % (w, test_count, T))
    print 'time: ', policy.ave_time(), 'acc: ', policy.accuracy

def czech_policy(w, count, T):
    cmd = '''./policy --inference Gibbs --policy cyclic_value --name test_policy/czech_w%d_tc%d_policy_T%d \
    --c 0.1 --T %d --numThreads 10 --model model/czech_gibbs_w%d.model --scoring Acc --windowL %d --testCount %d \
    --trainCount %d --verbose false --train data/czech_ner/train --test data/czech_ner/test --K 10''' \
    % (w, count, T, T, w,  w, count, count)
    print cmd
    os.system(cmd)  
    policy = PolicyResult('test_policy/czech_w%d_tc%d_policy_T%d' % (w, count, T))
    print 'time: ', policy.ave_time(), 'acc: ', policy.accuracy

farm = Farm()
for w in [0,1]:
  for T in [1,2,3,4]:
    farm.add('pos_ner/gibbs/toy/w%d/T%d' % (w, T), lambda w=w, T=T: pos_ner_gibbs(w, 100, T))
    farm.add('czech/gibbs/toy/w%d/T%d' % (w, T), lambda w=w, T=T: czech_gibbs(w, 100, T))
    farm.add('pos_ner/gibbs/full/w%d/T%d' % (w, T), lambda w=w, T=T: pos_ner_gibbs(w, 99999, T))
    farm.add('czech/gibbs/full/w%d/T%d' % (w, T), lambda w=w, T=T: czech_gibbs(w, 99999, T))
  for T in [25, 30, 35, 40, 45, 50]:
    farm.add('czech/policy/toy/w%d/T%d' % (w, T), lambda w=w, T=T: czech_policy(w, 100, T))
    farm.add('czech/policy/full/w%d/T%d' % (w, T), lambda w=w, T=T: czech_policy(w, 99999, T))
  for c in [0.0, 0.05, 0.1, 0.2, 0.3, 1]:
    farm.add('pos_ner/policy/full/w%d/c%f' % (w, c), lambda w=w, c=c: pos_ner_policy(w, 99999, c))
    farm.add('pos_ner/policy/toy/w%d/c%f' % (w, c), lambda w=w, c=c: pos_ner_policy(w, 100, c))

for f in [1,2,3,4]:
  for T in [1,2,3,4]:
    farm.add('ner/gibbs/toy/w2/f%d/T%d' % (f, T), lambda f=f, T=T: ner_gibbs(2, f, 100, T))
    farm.add('ner/gibbs/full/w2/f%d/T%d' % (f, T), lambda f=f, T=T: ner_gibbs(2, f, 99999, T))
  for c in [0.0, 0.05, 0.1, 0.2, 0.3, 1]:
    farm.add('ner/policy/toy/w2/f%d/c%f' % (f, c), lambda f=f, c=c: ner_policy(2, f, 100, c))
    farm.add('ner/policy/full/w2/f%d/c%f' % (f, c), lambda f=f, c=c: ner_policy(2, f, 99999, c))

if len(sys.argv) < 2:
  farm.visualize()
  exit(0)
farm.run(sys.argv[1])

"""
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
if sys.argv[1] == "wsj_cyclic_value":
  for c in c_l:
      cmd = '''./policy --inference Gibbs --policy cyclic_value \
      --name test_policy/wsj_cyclic_value_%f --c %f --numThreads 10 --eta 1 --K 10 \
      --model model/wsj_gibbs.model --train data/wsj/wsj-pos.train \
      --test data/wsj/wsj-pos.test ''' % (c, c)
      print cmd
      os.system(cmd)  
      policy = PolicyResult('test_policy/wsj_cyclic_value_%f' % c)
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
"""
