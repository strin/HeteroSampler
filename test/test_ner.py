import sys, os
from test import *
import unittest

class TestNER(unittest.TestCase):
    def test_small(self):
        cmd =  "./bin/tagging --T 8 --B 5 --train data/eng_ner/train_small --test data/eng_ner/test_small --eta 0.3 " + \
                        "--depthL 2 --windowL 0 --factorL 2 --output model/eng_ner/gibbs_small.model --scoring NER --Q 1 " + \
                        "--log 'log/eng_ner/small' --testFrequency 1"
        print cmd
        execute(cmd)
        line = execute("cat log/eng_ner/small | grep '<score>' -A 3 | tail -n 1")
        assert(float(line.split(' ')[-2]) >= 55)

    def test_small_gibbs_policy(self):
        cmd = "./bin/policy --type tagging --policy gibbs --output result/eng_ner/gibbs_small  " + \
                    "--model  model/eng_ner/gibbs_small.model --train  data/eng_ner/train_small " + \
                    "--test data/eng_ner/test_small --eta 1 --T 6   --log log/eng_ner/gibbs_small"
        print cmd
        execute(cmd)
        line = execute("cat result/eng_ner/gibbs_small/T4.xml | grep '<accuracy>' -A 1 | tail -n 1")
        assert(float(line) >= 0.5)

    def test_small_adaptive_policy(self):
            cmd = "./bin/policy --type tagging --policy adaptive --output result/eng_ner/adaptive_small  " + \
                        "--model  model/eng_ner/gibbs_small.model --train  data/eng_ner/train_small " + \
                        "--test data/eng_ner/test_small --eta 1 --T 6  " + \
                        "--feat 'sp cond-ent bias nb-vary nb-discord' --reward 0  --log log/eng_ner/adaptive_small"
            print cmd
            execute(cmd)
            line = execute("cat result/eng_ner/adaptive_small/b5.00.xml | grep '<accuracy>' -A 1 | tail -n 1")
            assert(float(line) >= 0.5)

if __name__ == "__main__":
    unittest.main()
