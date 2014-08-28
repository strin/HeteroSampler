from stat_policy import *
import re
import numpy as np
import matplotlib.pyplot as plt

def extract_number(s,notfound='NOT_FOUND'):
  regex=r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?'
  return [float(x) for x in re.findall(regex,s)]

def plot_pr(text, name, output):
  num = [] 
  for line in text.split('\n'):
    if line == '': 
      continue
    num.append(extract_number(line))
  num = np.array(num)
  plt.figure(num=None, figsize=(16, 8), dpi=100)
  plt.subplot(1,2,1)
  p, = plt.plot(num[:,0], num[:,1])
  plt.legend([p], [name])
  plt.title('prec/recall (sample)')
  plt.xlabel('precision')
  plt.ylabel('recall')
  plt.subplot(1,2,2) 
  p, = plt.plot(num[:,2], num[:,3])
  plt.legend([p], [name])
  plt.title('prec/recall (stop)')
  plt.xlabel('precision')
  plt.ylabel('recall')
  plt.savefig(output)
    
    
def plot_all(path_l, strategy_l, name_l, model, output):
  os.system('mkdir -p ' + output)
  acc_l = []
  time_l = []
  plot_l = []
  color_l=['r','g','b','k']
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
        os.system('mkdir - p ' + output + '/' + f)
        plot_pr(test.RH, name, output+'/'+f+'/pr_RH.png')
        plot_pr(test.RL, name, output+'/'+f+'/pr_RL.png')
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
    



if __name__ == '__main__':
  path_l = ['test_policy', 'test_policy/ner_roc', 'test_policy/unigram_roc']
  strategy_l = ['gibbs', 'multi_policy', 'multi_cyclic_value_unigram']
  name_l = ['Gibbs', 'Conditional Entropy', 'Unigram Entropy']
  model = 'ner_w2_f2_tc99999'
  output = 'result_policy/roc_w2_f2' 
  plot_all(path_l, strategy_l, name_l, model, output)

