import xml.etree.ElementTree as ElementTree
import numpy as np
import os, sys
import numpy.random as npr
import matplotlib.pyplot as plt

class StopResult:
  def __init__(me, name):
    me.name = name
    me.tree = ElementTree.parse(me.name+'/stop.xml')
    me.root = me.tree.getroot()
    for data in me.root:
      if data.tag == 'test':
        for item in data:
          if item.tag == 'accuracy':
            me.accuracy = float(item.text)
          elif item.tag == 'example':
            me.parse_test(item)
          elif item.tag == 'param':
            me.param = dict()
            for attr in item:
              if attr.tag == 'entry':
                me.param[attr.attrib['name']] = float(attr.attrib['value'])
          
  def parse_test(me, node):
    me.testex = list()
    for row in node:
      ex = dict()
      for attr in row:
        if attr.tag == 'dist' or attr.tag == 'time' \
          or attr.tag == 'resp':
          ex[attr.tag] = float(attr.text)
        else:
          ex[attr.tag] = attr.text
      me.testex.append(ex)

  def ave_time(me):
    time = list()
    for ex in me.testex:
      ex_len = len(ex['truth'].split('\t'))-1
      time.append(ex['time'] * ex_len)
    return np.mean(time)

  def plot_param(me):
    plt.figure(num=None, figsize=(15, 6), dpi=80, facecolor='w', edgecolor='k')
    ax = plt.subplot(111)
    keys = me.param.keys()
    values = me.param.values()
    plt.bar(range(len(keys)), values)
    ax.set_xticks(np.array(range(len(keys)))+0.35)
    ax.set_xticklabels(tuple(keys))
    for tick in ax.xaxis.get_major_ticks():
      tick.label.set_fontsize(10)
    plt.title('parameters')  
    plt.show()

if __name__ == '__main__':
  name = sys.argv[1]
  stop = StopResult(name)
  print stop.ave_time()
  stop.plot_param()
