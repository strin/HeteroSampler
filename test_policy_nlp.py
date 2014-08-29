import numpy as np
import os, sys
import numpy.random as npr
from farm import *
from stat_policy import *

thres_l = [0.5, 1, 1.5, 2, 2.5, 3, 1000]
T_l = [1, 2, 3, 4]
path = "test_policy"
run = lambda cmd : os.system('nlpsub -v ./test.sh "'+ cmd + '"')
toBool = lambda s : ("true" if s else "false")

def pos_ner_gibbs(w, test_count, T):
    cmd = '''./policy --inference Gibbs --policy gibbs --name '''+path+'''/pos_ner_w%d_tc%d_gibbs_T%d \
    --T %d --numThreads 5 --model model/ner_pos_gibbs_w%d.model --scoring Acc --windowL %d --testCount %d \
    --eta 1 --verbose false --train data/eng_pos_ner/train --test data/eng_pos_ner/test''' \
    % (w, test_count, T, T, w,  w, test_count)
    print cmd
    run(cmd)  
    policy = PolicyResult(path+'/pos_ner_w%d_tc%d_gibbs_T%d' % (w, test_count, T))
    print 'time: ', policy.ave_time(), 'acc: ', policy.accuracy

def pos_ner_gibbs_shared(w, test_count, T):
    cmd = '''./policy --inference Gibbs --policy gibbs_shared --name '''+path+'''/pos_ner_w%d_tc%d_gibbs \
    --T %d --numThreads 5 --model model/ner_pos_gibbs_w%d.model --scoring Acc --windowL %d --testCount %d \
    --eta 1 --verbose false --train data/eng_pos_ner/train --test data/eng_pos_ner/test''' \
    % (w, test_count, T, w,  w, test_count)
    print cmd
    run(cmd + ' &')  
    
def pos_ner_policy(w, count, c):
    cmd = '''./policy --inference Gibbs --policy cyclic_value --name '''+path+'''/pos_ner_w%d_tc%d_policy_c%f \
    --c %f --numThreads 5 --model model/ner_pos_gibbs_w%d.model --scoring Acc --windowL %d --testCount %d \
    --eta 1 --trainCount %d --verbose false --train data/eng_pos_ner/train --test data/eng_pos_ner/test''' \
    % (w, count, c, c, w,  w, count, count)
    print cmd
    run(cmd)  
    policy = PolicyResult(path+'/pos_ner_w%d_tc%d_policy_c%f' % (w, count, c))
    print 'time: ', policy.ave_time(), 'acc: ', policy.accuracy

def pos_ner_multi_policy_shared(w, count):
    cmd = '''./policy --inference Gibbs --policy multi_cyclic_value_shared --name '''+path+'''/pos_ner_w%d_tc%d_multi_policy \
    --K 1 --numThreads 10 --model model/ner_pos_gibbs_w%d.model --scoring Acc --windowL %d --testCount %d \
    --T 4 --eta 1 --trainCount %d --verbose false --train data/eng_pos_ner/train --test data/eng_pos_ner/test''' \
    % (w, count,  w,  w, count, count)
    print cmd
    run(cmd + ' &')  

def ner_gibbs_shared(w, f, test_count, T):
    cmd = '''./policy --inference Gibbs --policy gibbs_shared --name '''+path+'''/ner_w%d_f%d_tc%d_gibbs \
    --T %d --numThreads 10 --model model/ner_gibbs_w%d_d2_f%d.model --scoring NER --windowL %d --trainCount %d --testCount %d \
    --depthL 2 --factorL %d --verbose false --train data/eng_ner/train --test data/eng_ner/test''' \
    % (w, f, test_count, T,  w, f,  w, test_count, test_count, f)
    print cmd
    run(cmd)  

def ocr_gibbs_shared(f, test_count, T):
    cmd = '''./policy --inference Gibbs --policy gibbs_shared --name '''+path
    cmd += "/ocr_f%d_tc%d_gibbs " % (f, tc) 
    cmd += " --T %d --numThreads 10 --model model/ocr_f%d.model " % (T, f)
    cmd += " --scoring Acc --windowL 0 --depthL 0 --trainCount %d --testCount %d " % (test_count, test_count) \
    cmd += " --factorL %d --verbose false --train data/ocr/train0 --test data/ocr/test0 " % (f) \
    print cmd
    run(cmd)  

def ner_policy_shared(w, f, test_count):
    cmd = '''./policy --inference Gibbs --policy cyclic_value_shared --name '''+path+'''/ner_w%d_f%d_tc%d_policy \
    --K 1 --numThreads 10 --model model/ner_gibbs_w%d_d2_f%d.model --scoring NER --windowL %d --trainCount %d --testCount %d \
    --depthL 2 --factorL %d --verbose false --train data/eng_ner/train --test data/eng_ner/test''' \
    % (w, f, test_count,  w, f,  w, test_count, test_count, f)
    print cmd
    run(cmd)  

def ner_multi_policy_shared(w, f, test_count, notrain=False):
    cmd = '''./policy --inference Gibbs --policy multi_cyclic_value_shared --name '''+path+'''/ner_w%d_f%d_tc%d_multi_policy \
    --K 1 --numThreads 10 --model model/ner_gibbs_w%d_d2_f%d.model --scoring NER --windowL %d --trainCount %d --testCount %d \
    --T 4 --depthL 2 --factorL %d --verbose true --train data/eng_ner/train --test data/eng_ner/test --lets_notrain %s''' \
    % (w, f, test_count,  w, f,  w, test_count, test_count, f, toBool(notrain))
    print cmd
    run(cmd + ' &')  

def ocr_multi_policy_shared(f, test_count):
    cmd = '''./policy --inference Gibbs --policy multi_cyclic_value_shared --name '''+path
    cmd += "/ocr_f%d_tc%d_multi_policy " % (f, test_count)
    cmd += " --K 1 --numThreads 10 --model model/ocr_f%d.model " % (f)
    cmd += " --scoring Acc --windowL 0 --trainCount %d --testCount %d " % (test_count, test_count) 
    cmd += " --T 4 --depthL 0 --factorL %d --verbose false --train data/ocr/train0 --test data/ocr/test0 " % f
    cmd += " --dataset ocr "
    print cmd
    run(cmd + ' &')  

def ner_oracle_shared(w, f, test_count):
    cmd = '''./policy --inference Gibbs --policy cyclic_oracle_shared --name '''+path+'''/ner_w%d_f%d_tc%d_oracle \
    --K 1 --numThreads 10 --model model/ner_gibbs_w%d_d2_f%d.model --scoring NER --windowL %d --trainCount %d --testCount %d \
    --T 4 --depthL 2 --factorL %d --verbose false --train data/eng_ner/train --test data/eng_ner/test''' \
    % (w, f, test_count,  w, f,  w, test_count, test_count, f)
    print cmd
    run(cmd + ' &')  

def ner_multi_policy_unigram_shared(w, f, test_count, notrain=False):
    method = "multi_cyclic_value_unigram"
    cmd = '''./policy --inference Gibbs --policy '''+method+'''_shared --name '''+path
    cmd += '''/ner_w%d_f%d_tc%d_''' % (w, f, test_count) + method 
    cmd += ''' --K 1 --numThreads 10 --model model/ner_gibbs_w%d_d2_f%d.model ''' % (w, f)
    cmd += ''' --unigram_model model/ner_gibbs_w%d_d2_f1.model ''' % (w) 
    cmd += '''--scoring NER --windowL %d --trainCount %d --testCount %d ''' % (w, test_count, test_count)
    cmd += ''' --T 4 --depthL 2 --factorL %d --verbose true --train data/eng_ner/train --test data/eng_ner/test''' % (f)
    cmd += ''' --lets_notrain %s ''' % toBool(notrain)
    print cmd
    run(cmd + ' &')  

def ner_shared(w, f, test_count, method):
    cmd = '''./policy --inference Gibbs --policy '''+method+'''_shared --name '''+path
    cmd += '''/ner_w%d_f%d_tc%d_''' % (w, f, test_count) + method 
    cmd += ''' --K 1 --numThreads 10 --model model/ner_gibbs_w%d_d2_f%d.model ''' % (w, f)
    cmd += '''--scoring NER --windowL %d --trainCount %d --testCount %d ''' % (w, test_count, test_count)
    cmd += '''--T 4 --depthL 2 --factorL %d --verbose false --train data/eng_ner/train --test data/eng_ner/test''' % (f)
    print cmd
    run(cmd + ' &')  

def ner_gibbs(w, f, test_count, T):
    cmd = '''./policy --inference Gibbs --policy gibbs --name '''+path+'''/ner_w%d_f%d_tc%d_gibbs_T%d \
    --T %d --numThreads 5 --model model/ner_gibbs_w%d_d2_f%d.model --scoring NER --windowL %d --testCount %d \
    --depthL 2 --factorL %d --verbose false --train data/eng_ner/train --test data/eng_ner/test''' \
    % (w, f, test_count, T, T, w, f,  w, test_count, f)
    print cmd
    run(cmd)  
    policy = PolicyResult(path+'/ner_w%d_f%d_tc%d_gibbs_T%d' % (w, f, test_count, T))
    print 'time: ', policy.ave_time(), 'acc: ', policy.accuracy

def ner_policy(w, f, count, T):
    cmd = '''./policy --inference Gibbs --policy cyclic_value --name '''+path+'''/ner_w%d_f%d_tc%d_policy_T%f \
    --Tstar %f --numThreads 5 --model model/ner_gibbs_w%d_d2_f%d.model --scoring NER --windowL %d --testCount %d \
    --eta 1 --K 5 --depthL 2 --factorL %d --trainCount %d --verbose false --train data/eng_ner/train --test data/eng_ner/test''' \
    % (w, f, count, T, T, w, f,  w, count, f,  count)
    print cmd
    run(cmd)  
    policy = PolicyResult(path+'/ner_w%d_f%d_tc%d_policy_T%f' % (w, f, count, T))
    print 'time: ', policy.ave_time(), 'acc: ', policy.accuracy

def czech_gibbs_shared(w, test_count, T):
    cmd = '''./policy --inference Gibbs --policy gibbs_shared --name '''+path+'''/czech_w%d_tc%d_gibbs \
    --T %d --numThreads 10 --model model/czech_gibbs_w%d.model --scoring Acc --windowL %d --trainCount %d --testCount %d \
    --verbose false --train data/czech_ner/train --test data/czech_ner/test''' \
    % (w, test_count, T,  w,  w, test_count, test_count)
    print cmd
    run(cmd)  

def czech_policy_shared(w, test_count):
    cmd = '''./policy --inference Gibbs --policy cyclic_value_shared --name '''+path+'''/czech_w%d_tc%d_policy \
    --K 1 --numThreads 10 --model model/czech_gibbs_w%d.model --scoring Acc --windowL %d --trainCount %d --testCount %d \
    --verbose false --train data/czech_ner/train --test data/czech_ner/test''' \
    % (w, test_count,  w,  w, test_count, test_count)
    print cmd
    run(cmd)  

def czech_multi_policy_shared(w,  test_count, notrain=False):
    cmd = '''./policy --inference Gibbs --policy multi_cyclic_value_shared --name '''+path+'''/czech_w%d_tc%d_multi_policy \
    --K 1 --numThreads 10 --model model/czech_gibbs_w%d.model --scoring Acc --windowL %d --trainCount %d --testCount %d \
    --T 4 --verbose false --train data/czech_ner/train --test data/czech_ner/test --lets_notrain %s''' \
    % (w, test_count,  w, w, test_count, test_count, toBool(notrain))
    print cmd
    run(cmd)  


def czech_gibbs(w, test_count, T):
    cmd = '''./policy --inference Gibbs --policy gibbs --name '''+path+'''/czech_w%d_tc%d_gibbs_T%d \
    --T %d --numThreads 5 --model model/czech_gibbs_w%d.model --scoring Acc --windowL %d --testCount %d \
    --verbose false --train data/czech_ner/train --test data/czech_ner/test''' \
    % (w, test_count, T, T, w,  w, test_count)
    print cmd
    run(cmd)  
    policy = PolicyResult(path+'/czech_w%d_tc%d_gibbs_T%d' % (w, test_count, T))
    print 'time: ', policy.ave_time(), 'acc: ', policy.accuracy

def czech_policy(w, count, T):
    cmd = '''./policy --inference Gibbs --policy cyclic_value --name '''+path+'''/czech_w%d_tc%d_policy_T%f \
    --c 0.1 --T %f --numThreads 10 --model model/czech_gibbs_w%d.model --scoring Acc --windowL %d --testCount %d \
    --eta 1 --trainCount %d --verbose false --train data/czech_ner/train --test data/czech_ner/test --K 10''' \
    % (w, count, T, T, w,  w, count, count)
    print cmd
    run(cmd)  
    policy = PolicyResult(path+'/czech_w%d_tc%d_policy_T%f' % (w, count, T))
    print 'time: ', policy.ave_time(), 'acc: ', policy.accuracy

def wsj_gibbs_shared(w, f, count, T):
    cmd = "./policy --inference Gibbs --policy gibbs_shared --name "+path
    cmd += "/wsj_w%d_f%d_tc%d_gibbs " % (w, f, count)
    cmd += " --T %d --numThreads 10 --train data/wsj/wsj-pos.train --scoring Acc --testCount %d " % (T, count)
    cmd += " --test data/wsj/wsj-pos.test --model model/wsj_gibbs_w%d_d0_f%d.model --verbose false " % (w, f)
    cmd += " --windowL %d --depthL 0 --factorL %d " % (w, f)
    print cmd
    run(cmd + ' &')
  
def wsj_multi_policy_shared(w, f, count, T, notrain=False):
    cmd = '''./policy --inference Gibbs --policy multi_cyclic_value_shared --name '''
    cmd += path+"/wsj_w%d_f%d_tc%d_multi_policy " % (w, f, count)
    cmd += " --T %d --numThreads 10 --train data/wsj/wsj-pos.train --scoring Acc --testCount %d --trainCount %d" % (T, count ,count)
    cmd += " --eta 1 --test data/wsj/wsj-pos.test --model model/wsj_gibbs_w%d_d0_f%d.model --verbose false " % (w, f)
    cmd += " --K 1 --depthL 0 --windowL %d --factorL %d --lets_notrain %s" % (w, f, ("true" if notrain else "false"))
    print cmd
    run(cmd)

def wsj_multi_policy_unigram_shared(w, f, count, T, notrain=False):
    cmd = '''./policy --inference Gibbs --policy multi_cyclic_value_unigram_shared --name '''
    cmd += path+"/wsj_w%d_f%d_tc%d_multi_cyclic_value_unigram " % (w, f, count)
    cmd += " --T %d --numThreads 10 --train data/wsj/wsj-pos.train --scoring Acc --testCount %d --trainCount %d" % (T, count ,count)
    cmd += " --eta 1 --test data/wsj/wsj-pos.test --model model/wsj_gibbs_w%d_d0_f%d.model --verbose false " % (w, f)
    cmd += " --K 1 --depthL 0 --windowL %d --factorL %d --lets_notrain %s" % (w, f, ("true" if notrain else "false"))
    cmd += " --unigram_model model/wsj_gibbs_w%d_d0_f1.model " % w
    print cmd
    run(cmd)

TOY = 100
FULL = 99999

farm = Farm()

for w in [0,1,2]:
  for f in [1,2,3,4]:
    farm.add('wsj/gibbs/full/w%d/f%d' % (w, f), lambda w=w, f=f: wsj_gibbs_shared(w, f, FULL, 4))
    farm.add('wsj/multi_policy/full/w%d/f%d' % (w, f), lambda w=w, f=f: wsj_multi_policy_shared(w, f, FULL, 4))
    farm.add('wsj/multi_policy_unigram/full/w%d/f%d' % (w, f), lambda w=w, f=f: wsj_multi_policy_unigram_shared(w, f, FULL, 4))
    farm.add('wsj/gibbs/toy/w%d/f%d' % (w, f), lambda w=w, f=f: wsj_gibbs_shared(w, f, TOY, 4))
    farm.add('wsj/multi_policy/toy/w%d/f%d' % (w, f), lambda w=w, f=f: wsj_multi_policy_shared(w, f, TOY, 4))
    farm.add('wsj/multi_policy/notrain/w%d/f%d' % (w, f), lambda w=w, f=f: wsj_multi_policy_shared(w, f, FULL, 4, notrain=True))
    farm.add('wsj/multi_policy_unigram/notrain/w%d/f%d' % (w, f), lambda w=w, f=f: wsj_multi_policy_unigram_shared(w, f, FULL, 4, notrain=True))

for w in [0,1,2]:
  farm.add('czech/gibbs/w%d/toy'%w, lambda w=w: czech_gibbs_shared(w, TOY, 4)) 
  farm.add('czech/policy/w%d/toy'%w, lambda w=w: czech_policy_shared(w, TOY)) 
  farm.add('czech/gibbs/w%d/full'%w, lambda w=w: czech_gibbs_shared(w, FULL, 4)) 
  farm.add('czech/policy/w%d/full'%w, lambda w=w: czech_policy_shared(w, FULL)) 
  farm.add('czech/multi_policy/w%d/full'%w, lambda w=w: czech_multi_policy_shared(w, FULL)) 
  farm.add('czech/multi_policy/w%d/notrain'%w, lambda w=w: czech_multi_policy_shared(w, FULL, notrain=True)) 
  farm.add('czech/multi_policy/w%d/toy'%w, lambda w=w: czech_multi_policy_shared(w, TOY)) 
  farm.add('pos_ner/gibbs/w%d/full'%w, lambda w=w: pos_ner_gibbs_shared(w, FULL, 4)) 
  farm.add('pos_ner/multi_policy/w%d/full'%w, lambda w=w: pos_ner_multi_policy_shared(w, FULL)) 
  farm.add('pos_ner/multi_policy/w%d/toy'%w, lambda w=w: pos_ner_multi_policy_shared(w, TOY)) 

for f in [1,2,3,4]:
  for T in [1,2,3,4]:
    # farm.add('ner/gibbs/toy/w2/f%d/T%d' % (f, T), lambda f=f, T=T: ner_gibbs(2, f, 100, T))
    farm.add('ner/gibbs/full/w2/f%d/T%d' % (f, T), lambda f=f, T=T: ner_gibbs(2, f, FULL, T))
  farm.add('ner/gibbs/w2/toy/f%d'%f, lambda f=f: ner_gibbs_shared(2, f, TOY, 4)) 
  farm.add('ner/policy/w2/toy/f%d'%f, lambda f=f: ner_policy_shared(2, f, TOY)) 
  farm.add('ner/gibbs/w2/full/f%d'%f, lambda f=f: ner_gibbs_shared(2, f, FULL, 4)) 
  farm.add('ocr/gibbs/full/f%d'%f, lambda f=f: ocr_gibbs_shared(f, FULL, 4)) 
  farm.add('ner/policy/w2/full/f%d'%f, lambda f=f: ner_policy_shared(2, f, FULL)) 
  farm.add('ner/oracle/w2/full/f%d'%f, lambda f=f: ner_oracle_shared(2, f, FULL)) 
  farm.add('ner/multi_policy_unigram/w2/full/f%d'%f, lambda f=f: ner_multi_policy_unigram_shared(2, f, FULL)) 
  farm.add('ner/multi_policy_unigram/w2/notrain/f%d'%f, lambda f=f: ner_multi_policy_unigram_shared(2, f, FULL, notrain=True)) 
  farm.add('ner/multi_policy/w2/full/f%d'%f, lambda f=f: ner_multi_policy_shared(2, f, FULL)) 
  farm.add('ner/multi_policy/notrain/w2/f%d'%f, lambda f=f: ner_multi_policy_shared(2, f, FULL, notrain=True)) 
  farm.add('ner/multi_policy/w2/toy/f%d'%f, lambda f=f: ner_multi_policy_shared(2, f, TOY)) 
  farm.add('ocr/multi_policy/full/f%d'%f, lambda f=f: ocr_multi_policy_shared(f, FULL)) 
  farm.add('ocr/multi_policy/toy/f%d'%f, lambda f=f: ocr_multi_policy_shared(f, TOY)) 


  '''
  for T in [1.0, 1.25, 1.5, 1.75, 2]:
    farm.add('ner/policy/toy/w2/f%d/T%f' % (f, T), lambda f=f, T=T: ner_policy(2, f, TOY, T))
    farm.add('ner/policy/full/w2/f%d/T%f' % (f, T), lambda f=f, c=c: ner_policy(2, f, FULL, T))
  '''
if len(sys.argv) < 2:
  farm.visualize()
  exit(0)
if sys.argv[1][-1] == ':':
  farm.visualize(farm.find(sys.argv[1][:-1]))
  exit(0)
if len(sys.argv) >= 3:
  path = sys.argv[2]
  run('mkdir -p '+path)
farm.run(sys.argv[1])

"""
if sys.argv[1] == "toy_entropy":
    for thres in thres_l:
      cmd = '''./policy --inference Gibbs --policy entropy --name test_policy/entropy_%0.2f \
      --testCount 100 --threshold %0.2f --numThreads 10''' % (thres, thres)
      print cmd
      run(cmd)  
      policy = PolicyResult('test_policy/entropy_%0.2f' % thres)
      print 'time: ', policy.ave_time(), 'acc: ', policy.accuracy
if sys.argv[1] == "toy_cyclic":
  for c in c_l:
      cmd = '''./policy --inference Gibbs --policy cyclic --name test_policy/cyclic_%f \
      --trainCount 100 --testCount 100 --c %f --numThreads 10 --eta 1 --K 5''' % (c, c)
      print cmd
      run(cmd)  
      policy = PolicyResult('test_policy/cyclic_%f' % c)
      print 'time: ', policy.ave_time(), 'acc: ', policy.accuracy
elif sys.argv[1] == "toy_gibbs":
  for T in T_l:
    cmd = '''./policy --inference Gibbs --policy gibbs --name test_policy/gibbs_T%d --T %d --testCount 100 --numThreads 4''' % (T, T)
    print cmd
    run(cmd)  
    policy = PolicyResult('test_policy/gibbs_T%d' % T)
    print 'time: ', policy.ave_time(), 'acc: ', policy.accuracy
if sys.argv[1] == "wsj_entropy":
    for thres in thres_l:
      cmd = '''./policy --inference Gibbs --policy entropy \
      --name test_policy/wsj_entropy_%0.2f --train data/wsj/wsj-pos.train \
      --test data/wsj/wsj-pos.test --threshold %0.2f --numThreads 10 \
      --model model/wsj_gibbs.model ''' % (thres, thres)
      print cmd
      run(cmd)  
      policy = PolicyResult('test_policy/wsj_entropy_%0.2f' % thres)
      print 'time: ', policy.ave_time(), 'acc: ', policy.accuracy
if sys.argv[1] == "wsj_cyclic":
  for c in c_l:
      cmd = '''./policy --inference Gibbs --policy cyclic \
      --name test_policy/wsj_cyclic_%f --train data/wsj/wsj-pos.train \
      --test data/wsj/wsj-pos.test --c %f --numThreads 10 --eta 1 --K 10 \
      --model model/wsj_gibbs.model''' % (c, c)
      print cmd
      run(cmd)  
      policy = PolicyResult('test_policy/wsj_cyclic_%f' % c)
      print 'time: ', policy.ave_time(), 'acc: ', policy.accuracy
if sys.argv[1] == "wsj_cyclic_value":
  for c in c_l:
      cmd = '''./policy --inference Gibbs --policy cyclic_value \
      --name test_policy/wsj_cyclic_value_%f --c %f --numThreads 10 --eta 1 --K 10 \
      --model model/wsj_gibbs.model --train data/wsj/wsj-pos.train \
      --test data/wsj/wsj-pos.test ''' % (c, c)
      print cmd
      run(cmd)  
      policy = PolicyResult('test_policy/wsj_cyclic_value_%f' % c)
      print 'time: ', policy.ave_time(), 'acc: ', policy.accuracy
elif sys.argv[1] == "wsj_gibbs":
  for T in T_l:
    cmd = '''./policy --inference Gibbs --policy gibbs --name test_policy/wsj_gibbs_T%d \
    --T %d --numThreads 10 --train data/wsj/wsj-pos.train \
    --test data/wsj/wsj-pos.test --model model/wsj_gibbs.model ''' % (T, T)
    print cmd
    run(cmd)  
    policy = PolicyResult('test_policy/wsj_gibbs_T%d' % T)
    print 'time: ', policy.ave_time(), 'acc: ', policy.accuracy
if sys.argv[1] == "ner_entropy":
    for thres in thres_l:
      cmd = '''./policy --inference Gibbs --windowL 2 --policy entropy \
      --name test_policy/ner_entropy_%0.2f --threshold %0.2f --numThreads 10 \
      --model model/ner_gibbs.model --mode NER''' % (thres, thres)
      print cmd
      run(cmd)  
      policy = PolicyResult('test_policy/ner_entropy_%0.2f' % thres)
      print 'time: ', policy.ave_time(), 'acc: ', policy.accuracy
if sys.argv[1] == "ner_cyclic":
  for c in c_l:
      cmd = '''./policy --inference Gibbs --policy cyclic \
      --name test_policy/ner_cyclic_%f --c %f --numThreads 10 --eta 1 --K 10 \
      --model model/ner_gibbs.model --windowL 2 --mode NER ''' % (c, c)
      print cmd
      run(cmd)  
      policy = PolicyResult('test_policy/ner_cyclic_%f' % c)
      print 'time: ', policy.ave_time(), 'acc: ', policy.accuracy
if sys.argv[1] == "ner_cyclic_value":
  for c in c_l:
      cmd = '''./policy --inference Gibbs --policy cyclic_value \
      --name test_policy/ner_cyclic_value_%f --c %f --numThreads 10 --eta 1 --K 10 \
      --model model/ner_gibbs.model --windowL 2 --mode NER ''' % (c, c)
      print cmd
      run(cmd)  
      policy = PolicyResult('test_policy/ner_cyclic_value_%f' % c)
      print 'time: ', policy.ave_time(), 'acc: ', policy.accuracy
elif sys.argv[1] == "ner_gibbs":
  for T in T_l:
    cmd = '''./policy --inference Gibbs --policy gibbs --name test_policy/ner_gibbs_T%d \
    --T %d --numThreads 10 --model model/ner_gibbs.model --mode NER --windowL 2''' % (T, T)
    print cmd
    run(cmd)  
    policy = PolicyResult('test_policy/ner_gibbs_T%d' % T)
    print 'time: ', policy.ave_time(), 'acc: ', policy.accuracy
elif sys.argv[1] == "toy0":
  for T in T_l:
    cmd = '''./stop --inference Gibbs --T %d --name gibbs0_T%d --numThreads 10 \
    --trainCount 1000 --testCount 1000 --adaptive false''' % (T, T)
    print cmd
    run(cmd)  
    stop = StopResult('gibbs0_T%d' % T)
    print 'T = ', T, 'acc = ', stop.accuracy
"""
