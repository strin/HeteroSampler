#!/bin/sh
# learn model using ./pos, $1: task, $2: inference method.
if [ $1 == "POS_NER" ]; then 
  cmd="./pos --inference $2 --T 10 --B 3 --train data/eng_pos_ner/train --test data/eng_pos_ner/test --eta 1 --windowL 2 \
    --output model/ner_pos_gibbs.model --scoring Acc "
  echo $cmd
  ($cmd) > ner_pos_gibbs.xml
fi

