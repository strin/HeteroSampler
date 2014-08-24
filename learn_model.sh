#!/bin/bash
# learn model using ./pos, $1: task, $2: inference method.
if [ $1 == "NER" ]; then 
  factorL=$4
  cmd="./pos --inference $2 --T 1 --B 3 --train data/eng_ner/train --test data/eng_ner/test --eta 10 \
    --depthL 2 --windowL "$3" --factorL "$factorL" --output model/ner_gibbs_w"$3"_d2_f"$factorL".model --scoring NER --Q 5"
  echo $cmd
  # ($cmd) > ner_gibbs_w$3_d2_f$factorL.xml & 
elif [ $1 == "POS_NER" ]; then 
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
      --output model/czech_gibbs_w"$windowL".model --scoring Acc --Q 3 "
    echo $cmd
    ($cmd) > czech_gibbs_w$windowL.xml &
  done
else
  echo "1 = [task: NER, Czech]"
  echo "2 = [inference method: Gibbs]"
  echo "3 = windowL"
  echo "4 = factorL"
fi

