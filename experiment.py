import os, sys
import numpy as np

class Experiment:
  def __check_empty__(self, line):
    return line == ""

  def __check_tag_start__(self, line):
    return line[0] == '<' and line[-1] == '>' and line[1] != '/'

  def __check_tag_end__(self, line):
    return line[0] == '<' and line[1] == '/'

  def stack_str(self, stack):
    res = ''
    for line in stack:
      res = res+line+'/'
    return res
    
  def parse(self, path):
    f = open(path, 'r')
    stack = list()
    self.result = dict()
    self.acc = dict()
    while True:
      line = f.readline()
      line = line.replace('\n', '')
      if self.__check_empty__(line):
        break
      if self.__check_tag_start__(line):
        stack.append(line)
      elif self.__check_tag_end__(line):
        stack.pop()
      else:
        if len(stack) > 1 and stack[1] == '<test_lag>':
          self.test_lag = float(line)
        if len(stack) > 2 and stack[2] == '<test>':
          if len(stack) > 4 and stack[4] == '<truth>':
            item = {'truth':line}
          elif len(stack) > 4 and stack[4] == '<tag>':
            item['tag'] = line  
          elif len(stack) > 4 and stack[4] == '<dist>':
            item['dist'] = float(line)
            self.result[stack[3]] = item
        elif len(stack) > 2 and stack[2] == '<score>':
          self.acc.append(float(line.split(' ')[3]))
    return self.result


def analyze_infer():
  expr = Experiment() 
  result1 = expr.parse('wsj_simple.xml')
  result2 = expr.parse('wsj_gibbs.xml')
  score = list()
  for key in result1.keys():
    score.append((key, result1[key]['dist']-result2[key]['dist']))
  score = sorted(score, key=lambda x : x[1], reverse=True)
  for i in range(100):
    key = score[i][0]
    print '--- Ground Truth ---'
    print result1[key]['truth']
    print '--- Independent Inference ---'
    print result1[key]['tag']
    print '[errors = ', result1[key]['dist'], ']'
    print '--- Full Inference ---'
    print result2[key]['tag']
    print '[errors = ', result2[key]['dist'], ']'
    print 

def plot_acc():
  expr = Experiment()
  expr.parse('wsj_simple.xml')
  plt.plot(np.array(range(len(expr.acc)))*expr.test_lag, expr.acc)
  plt.show()

if __name__ == '__main__':
  plot_acc()

      
