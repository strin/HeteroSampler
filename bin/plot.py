import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import *
import sys, os
import subprocess


def execute(cmd):
    ps = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = ps.communicate()[0]
    return output


def linesToArray(lines):
    lines = lines.split('\n')
    return [float(line) for line in lines if line != '']


def loadAcc(path):
    acc = linesToArray(execute("for file in `ls -t %s`; do cat $file | grep '<accuracy>' -A 1 | tail -n 1; done" % path))
    time = linesToArray(execute("for file in `ls -t %s`; do cat $file | grep '<time>' -A 1 | tail -n 1; done" % path))
    (time, acc) = zip(*sorted(zip(time, acc), key=lambda x: x[0]))
    return (list(time), list(acc))


def loadAccWallclock(path):
    acc = linesToArray(execute("for file in `ls -t %s`; do cat $file | grep '<accuracy>' -A 1 | tail -n 1; done" % path))
    time = linesToArray(execute("for file in `ls -t %s`; do cat $file | grep '<wallclock>' -A 1 | tail -n 1; done" % path))
    (time, acc) = zip(*sorted(zip(time, acc), key=lambda x: x[0]))
    return (list(time), list(acc))


def loadWallclock(path):
    time = linesToArray(execute("for file in `ls -t %s`; do cat $file | grep '<wallclock>' -A 1 | tail -n 1; done" % path))
    sample = linesToArray(execute("for file in `ls -t %s`; do cat $file | grep '<wallclock_sample>' -A 1 | tail -n 1; done" % path))
    policy = linesToArray(execute("for file in `ls -t %s`; do cat $file | grep '<wallclock_policy>' -A 1 | tail -n 1; done" % path))
    return (list(time), list(sample), list(policy))


import matplotlib.font_manager as fm
fm.FontProperties(family='font.serif')

def plotPerformanceWithBudgets(paths, output, _title, _xlabel, _ylabel,
                                _linewidth = 3.0,
                                _labelsize = 30,
                                _legendsize = 25,
                                _ticksize = 25,
                                _markersize = 0,
                                time_scale = 1,
                                acc_scale = 1):
    h = figure(num=None, figsize=(12, 10), dpi=80)
    (time, acc) = loadAcc(paths[0])
    plot(np.array(time) / time_scale, np.array(acc) / acc_scale, color='r', marker='s', markersize=_markersize, linewidth=_linewidth)
    (time, acc) = loadAcc(paths[1])
    plot(np.array(time) / time_scale, np.array(acc) / acc_scale, color='b', marker='^', markersize=_markersize, linewidth=_linewidth)
    title(_title, fontsize=_labelsize)
    xlabel(_xlabel, fontsize=_labelsize)
    ylabel(_ylabel, fontsize=_labelsize)
    legend(['All features', 'baseline'], loc=4, fontsize=_legendsize)
    tick_params(axis='both', which='major', labelsize=_ticksize)
    savefig(output)
    return h


def plotPolicyOverhead(path,  output, _title, _xlabel, _ylabel, 
                                _linewidth = 3.0,
                                _labelsize = 30,
                                _legendsize = 25,
                                _ticksize = 25,
                                _markersize = 5):
    figure(num=None, figsize=(12, 10), dpi=80)
    (time, sample, policy) = loadWallclock(path)
    plot(time, np.array(policy), color='b', linewidth=_linewidth)
    plot(time, np.array(sample)+np.array(policy), color='r', linewidth=_linewidth, fillstyle='bottom')
    fill_between(time,np.array(sample)+np.array(policy),0,color='#FAA2A2')
    fill_between(time,np.array(policy),0,color='#ACBBFA')
    title(_title, fontsize=_labelsize)
    xlabel(_xlabel, fontsize=_labelsize)
    ylabel(_ylabel, fontsize=_labelsize)
    legend(['Policy', 'Overall'], loc=2, fontsize=_legendsize)
    tick_params(axis='both', which='major', labelsize=_ticksize)
    savefig(output)


def sysopen(path):
    if sys.platform == 'darwin':
        os.system('open ' + path)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print 'usage: python plot.py [which task=ner/wsj/ocr/...] [type of figure:budget/overhead]'
        exit(0)
    task = sys.argv[1]
    typefig = "budget"
    if len(sys.argv) > 2:
        typefig = sys.argv[2]
    if len(sys.argv) > 3:
        path_gibbs = sys.argv[3] + '/'
    else:
        path_gibbs = './'
    if len(sys.argv) > 4:
        path_policy = sys.argv[4] + '/'
    else:
        path_policy = './'
    if len(sys.argv) > 5:
        path_output = sys.argv[5] + '/'
    else:
        path_output = './'

    print 'path_gibbs = ', path_gibbs
    print 'path_policy = ', path_policy
    print 'path_output = ', path_output

    if task == 'ner':
        os.system('mkdir -p ' + path_output + 'result/eng_ner/')
        if typefig == 'budget':
            h = plotPerformanceWithBudgets(paths=[path_policy+'result/eng_ner/adaptive/b*.xml',
                                          path_gibbs+'result/eng_ner/gibbs/T*.xml'],
                                   output=path_output+'result/eng_ner/budget.png',
                                   _title='Performance vs. Budget on NER',
                                   _xlabel='Average Number of  Transations',
                                   _ylabel='F1 score')
            sysopen(path_output+'result/eng_ner/budget.png')
        elif typefig == 'overhead':
            plotPolicyOverhead(path=path_policy+'result/eng_ner/adaptive/b*.xml',
                               output=path_output+'result/eng_ner/overhead.png',
                                   _title='', 
                                   _xlabel='Average Number of Transitions', 
                                   _ylabel='Wall-clock Seconds')
            sysopen(path_output+'result/eng_ner/overhead.png')



