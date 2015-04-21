import subprocess

def execute(cmd):
  ps = subprocess.Popen(cmd, shell=True,stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  output = ps.communicate()[0]
  return output
