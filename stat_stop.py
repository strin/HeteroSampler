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
      time.append(ex['time'] * ex_len)
      # time.append(ex['time'])
    # print time
    return np.mean(time)

  def ave_feat(me):
    keys = me.testex[0]['feat'].keys()
    values = list()
    for ex in me.testex:
      values.append(ex['feat'].values())
    return zip(keys, np.mean(values, 0))
      

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

  # compare with you, see if you improve my inference.
  def compare(me, you):
    if len(me.testex) != len(you.testex): 
      return None
    hit = 0
    allpred = 0
    for (ex, ey) in zip(me.testex, you.testex):
      ex_len = len(ex['truth'].split('\t'))-1
      allpred += ex_len
      if ex['dist'] <= ey['dist']: # inference not improved.
        hit += ex_len-ex['dist']
        print ex['truth']
        print ex['tag']
        print ey['tag']
        print
      else:
        hit += ex_len-ey['dist']
        #print ey['truth']
        #print ey['tag']
        #print
    return (hit/float(allpred), html)

  # compare with you, and visualize the result.
  def viscomp(me, you):
    if len(me.testex) != len(you.testex):
      return None
    head = "<style>"
    head += '''td {
      background-color: #FFFFFF;
      transition: background-color 0.3s, padding-top 0.1s;
      padding-top: 5px;
    }
    td:hover{
      background-color: #EDD861;
      padding-top: 0px;
    }'''
    head += '</style>'
    body = ""
    count = 0
    for (ex, ey) in zip(me.testex, you.testex):
      token_truth = ex['truth'].replace('\n', '').split('\t')
      token0 = ex['tag'].replace('\n', '').split('\t')
      token1 = ey['tag'].replace('\n', '').split('\t')
      words = [token.split('/')[0] for token in token_truth if token != '']
      true_tag = [token.split('/')[1] for token in token_truth if token != '']
      tag0 = [token.split('/')[1] for token in token0 if token != ''] 
      tag1 = [token.split('/')[1] for token in token1 if token != '']
      
      body += "<p><table><tr>"
      body += '''<td style='background-color: #000000; color: #ffffff'>%d</td>''' % count
      body += '''<td style='text-align: center; color: #9C9C9C; font-size: 14'><b style='font-size: 16'> Words</b> <br>
      Truth <br> Pass 0 <br> Pass 1 </td>'''
      for (w, t, t0, t1) in zip(words, true_tag, tag0, tag1):
        c0 = '#000000'
        c1 = '#000000'
        if t0 == t and t1 != t:
          c0 = '11B502'
          c1 = 'ED2143'
        elif t0 != t and t1 == t:
          c0 = 'ED2143'
          c1 = '11B502'
        elif t0 != t and t1 != t:
          c0 = c1 = 'ED2143'
          
        body += '''<td style='text-align: center; font-size: 14'> <b style='font-size: 16'> %s </b> <br>  
        %s <br> <span style='color: %s'> %s </span> <br> <span style='color: %s'> %s 
        </span> </td>''' % (w, t, c0, t0, c1, t1)
      body += "</tr></table></p>"
      count += 1
    return '''<html>\n<head>\n%s</head>\n<body>\n%s</body>\n</html>''' % (head, body)

if __name__ == '__main__':
  name = sys.argv[1]
  if name == '__compare__':
    stop0 = StopResult('test_stop/gibbs0_T0')
    stop1 = StopResult('test_stop/gibbs0_T1')
    acc = stop0.compare(stop1)
    print 'acc = ', acc
  elif name == '__viscomp__':
    stop0 = StopResult('test_stop/gibbs0_T0')
    stop1 = StopResult('test_stop/gibbs0_T1')
    print stop0.viscomp(stop1)
  else:
    stop = StopResult(name)
    print stop.ave_time()
    print stop.ave_feat()
    stop.plot_param()
