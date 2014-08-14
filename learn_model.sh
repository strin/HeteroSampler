#!/bin/bash
# learn model using ./pos, $1: task, $2: inference method.
if [ $1 == "POS_NER" ]; then 
  for windowL in `seq 0 2` 
  do
    cmd="./pos --inference $2 --T 10 --B 3 --train data/eng_pos_ner/train --test data/eng_pos_ner/test --eta 1 --windowL "$windowL" \
      --output model/ner_pos_gibbs_w"$windowL".model --scoring Acc --Q 2"
    echo $cmd
    ($cmd) > pos_ner_gibbs_w$windowL.xml &
  done
elif [ $1 == "Czech" ]; then 
  for windowL in `seq 0 2`
  do
    cmd="./pos --inference $2 --T 10 --B 3 --train data/czech_ner/train --test data/czech_ner/test --eta 1 --windowL "$windowL" \
      --output model/ner_pos_gibbs_w"$windowL".model --scoring NER --Q 3 "
    echo $cmd
    ($cmd) > czech_gibbs_$windowL.xml &
  done
fi

