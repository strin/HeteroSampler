from stat_policy import *
import re
import numpy as np
import matplotlib.pyplot as plt

color_l=['r','g','b','k']

def plot_pr(fig, pathi, text, name):
  num = [] 
  for line in text.split('\n'):
    if line == '': 
      continue
    if line.find('nan') == -1:
      num.append(extract_number(line))
  num = np.array(num)
  plt.figure(num=fig, figsize=(16, 16), dpi=100)
  plt.subplot(2,2,1);
  p, = plt.plot(num[:,0], num[:,2], '%s-' % color_l[pathi])
  plt.title('recall (sample)')
  plt.xlabel('threshold')
  plt.ylabel('recall')
  plt.subplot(2,2,2);
  p, = plt.plot(num[:,0], num[:,4], '%s-' % color_l[pathi])
  plt.title('recall (stop)')
  plt.xlabel('threshold')
  plt.ylabel('recall')
  plt.subplot(2,2,3)
  p, = plt.plot(num[:,1], num[:,2], '%s-' % color_l[pathi])
  plt.title('prec/recall (sample)')
  plt.xlabel('precision')
  plt.ylabel('recall')
  plt.subplot(2,2,4) 
  p, = plt.plot(num[:,3], num[:,4])
  plt.title('prec/recall (stop)')
  plt.xlabel('precision')
  plt.ylabel('recall')
    
    
def plot_all(path_l, strategy_l, name_l, model, output):
  os.system('mkdir -p ' + output)
  acc_l = []
  time_l = []
  plot_l = []
  for (pathi, name, stg, path) in zip(range(len(name_l)), name_l, strategy_l, path_l):
    files = os.listdir(path)
    files = [f[0] for f in sorted([(f, os.stat(path+'/'+f)) for f in files], key=lambda x: x[1].st_ctime)]
    files = [f for f in files if f.find(model+'_'+stg) == 0 and f != model+'_'+stg]
    acc = []
    time = []
    for f in files:
      print f
      if f.find('_train') != -1:
        test = PolicyResultLite(path+'/'+f+'/policy.xml') 
        print test.RH
        plot_pr(1, pathi, test.RH, name)
        plot_pr(2, pathi, test.RL, name)
      else:
        test = PolicyResultLite(path+'/'+f+'/policy.xml')
        acc.append(test.acc) 
        time.append(test.time)
    plt.figure(num=-1, figsize=(16, 8), dpi=100)
    p, = plt.plot(time, acc, '%s-' % (color_l[pathi]))
    plot_l.append(p)
    acc_l.append(acc)
    time_l.append(time)
  plt.figure(num=-1)
  plt.legend(plot_l, name_l, loc=4)
  plt.savefig(output+'/main.png')
  plt.show()
  plt.figure(1)
  plt.savefig(output+'/RH.png')
  plt.figure(2)
  plt.savefig(output+'/RL.png')
    



if __name__ == '__main__':
  mode = 'temp'
  if mode == 'normal':
    path_l = ['test_policy/oracle_roc', 'test_policy', 'test_policy/policy_roc_T2', 'test_policy/unigram_roc_fixed']
    strategy_l = ['multi_policy', 'gibbs', 'multi_policy', 'multi_cyclic_value_unigram']
    name_l = ['Oracle', 'Gibbs', 'Conditional Entropy', 'Unigram Entropy']
    model = 'ner_w2_f2_tc99999'
    output = 'result_policy/roc_w2_f2' 
    plot_all(path_l, strategy_l, name_l, model, output)
  elif mode == 'train':
    path_l = ['test_policy', 'test_policy/policy_roc', 'test_policy/policy_notrain_roc']
    strategy_l = ['gibbs', 'multi_policy', 'multi_policy']
    name_l = ['Gibbs', 'Policy - Train', 'Policy - No Train']
    model = 'ner_w2_f2_tc99999'
    output = 'result_policy/policy_train' 
    plot_all(path_l, strategy_l, name_l, model, output)
  elif mode == 'temp':
    path_l = ['test_policy/temp_gibbs', 'test_policy/temp', 'test_policy/temp_unigram']
    strategy_l = ['gibbs', 'multi_policy', 'multi_cyclic_value_unigram']
    name_l = ['Gibbs', 'Conditional Entropy', 'Unigram Entropy']
    model = 'ner_w2_f2_tc99'
    output = 'result_policy/temp' 
    plot_all(path_l, strategy_l, name_l, model, output)

