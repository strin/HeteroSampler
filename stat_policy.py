import xml.etree.ElementTree as ElementTree
import numpy as np
import os, sys
import numpy.random as npr
import matplotlib.pyplot as plt

class PolicyResult:
  def __init__(me, name):
    me.name = name
    me.tree = ElementTree.parse(me.name+'/policy.xml')
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
        elif attr.tag == 'feat':
          ex['feat'] = dict()
          for entry in attr:
            ex['feat'][entry.attrib['name']] = float(entry.attrib['value'])
        else:
          ex[attr.tag] = attr.text
      me.testex.append(ex)

  def ave_time(me):
    time = list()
    for ex in me.testex:
      ex_len = len(ex['truth'].split('\t'))-1
      # time.append(ex['time'] * ex_len)
      time.append(ex['time'])
    # print time
    return np.mean(time)


if __name__ == '__main__':
  name = sys.argv[1]
  if name == '__compare__':
    stop0 = PolicyResult('gibbs0_T0')
    stop1 = PolicyResult('gibbs0_T1')
    acc = stop0.compare(stop1)
    print 'acc = ', acc
  else:
    stop = PolicyResult(name)
    print 'time = ', stop.ave_time(), 'accuracy = ', stop.accuracy
