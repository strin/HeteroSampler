import sys, os
from test import *
import unittest

class TestNER(unittest.TestCase):
    def test_small(self):
        os.system("""./tagging --T 8 --B 5 --train data/eng_ner/train_small --test data/eng_ner/test_small --eta 0.3
                                        --depthL 2 --windowL 0 --factorL 2 --output model/eng_ner/gibbs_small.model --scoring NER --Q 1
                                         --log "log/eng_ner/small" --testFrequency 1
                        """)
        line = execute("cat log/eng_ner/small | grep '<score>' -A 3 | tail -n 1")
        assert(float(line.split(' ')[-2]) >= 55)

if __name__ == "__main__":
    unittest.main()
