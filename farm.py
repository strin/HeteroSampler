import os, sys

# farm is implemented using hierarchical dict.
class Farm: 
  def __init__(me):
    me.root = dict()

  # fine func according to path.
  def find(me, path):
    path_l = path.split('/')
    node = me.root
    for p in path_l:
      if p == '': 
        continue
      if not isinstance(node, dict) or not node.has_key(p):
        return dict()
      else:
        node = node[p]
    return node

  # add a func to path.
  def add(me, path, func):
    path_l = path.split('/')
    node = me.root
    for p in path_l:
      if p == '':
        continue
      if not node.has_key(p):
        node[p] = dict()
      node = node[p]
    node['_'] = func
  
  # list all paths.
  def list(me, path_l=list(), node=None):
    if node == None:
      node = me.root
    str_l = list()
    for key in node.keys():
      if key == '_':
        str_l.append('/'.join(path_l)+'/'+str(node['_']))
      else:
        str_l += me.list(path_l+[key], node[key])
    return str_l

  # nicely visualize all paths.
  def visualize(me, node=None, indent=0):
    if node == None:
      node = me.root
    if node.has_key('_'):
      print ''.join(['  ']*indent), node['_']
    for key in node.keys():
      if key == '_':
        continue
      else:
        print ''.join(['  ']*indent), key+'/'
        me.visualize(node[key], indent+1)

  # report all paths.
  def __str__(me):
    return '\n'.join(me.list())

  # execute recursively from a node.
  def run_recursive(me, node):
    for key in node.keys():
      if key == '_':
        node[key]();
      else:
        me.run_recursive(node[key])

  # execute a path.
  def run(me, path=''):
    me.run_recursive(me.find(path))
  
if __name__ == '__main__':
  def task_add(a, b):
    print 'add', a, b, '=', a + b

  def task_multiply(a, b):
    print 'mul', a, b, '=', a * b;
  
  farm = Farm()
  farm.add('add/1,2', lambda: task_add(1,2))
  farm.add('add/-1,1/', lambda: task_add(-1,1))
  farm.add('mul/32,32', lambda: task_multiply(32,32))

  print farm
  farm.run()





      
    

