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
      ex['feat'] = list()
      for attr in row:
        if attr.tag == 'dist' or attr.tag == 'time':
          ex[attr.tag] = float(attr.text)
        elif attr.tag == 'feat':
          feat = dict()
          for entry in attr:
            feat[entry.attrib['name']] = float(entry.attrib['value'])
          ex['feat'].append(feat)
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

  # compare with you, and visualize the result.
  def viscomp(me, you):
    if len(me.testex) != len(you.testex):
      return None
    head = "<style>"
    head += '''td {
      transition: padding-top 0.1s;
      padding-top: 5px;
    }
    td:hover {
      padding-top: 0px;
    }
    td span: {
      transition: background-color 0.3s; 
      background-color: #FFFFFF;
    }
    td span:hover{
      background-color: #EDD861;
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
      for (i, (w, t, t0, t1)) in enumerate(zip(words, true_tag, tag0, tag1)):
        f0 = ex['feat'][i]
        f1 = ey['feat'][i]
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
        %s <br> <span title=\"%s\" style='color: %s'> %s </span> <br> <span title=\"%s\" style='color: %s'> %s 
        </span> </td>''' % (w, t, str(f0), c0, t0, str(f1), c1, t1)
      body += "</tr></table></p>"
      count += 1
    return '''<html>\n<head>\n%s</head>\n<body>\n%s</body>\n</html>''' % (head, body)

if __name__ == '__main__':
  name = sys.argv[1]
  if name == '__viscomp__':
    policy0 = PolicyResult('test_policy/gibbs_T1')
    policy1 = PolicyResult('test_policy/gibbs_T2')
    print policy0.viscomp(policy1)
  else:
    stop = PolicyResult(name)
    print 'time = ', stop.ave_time(), 'accuracy = ', stop.accuracy
