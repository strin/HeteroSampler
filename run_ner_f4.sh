if [ -z "$Q" ]; then
  Q=1
fi
feat="sp cond-ent  bias nb-vary nb 01-cond-ent"
./policy --inference Gibbs --policy blockpolicy --name test/ner/w2/f4/blockpolicy/$name/tcall --K 1 \
--numThreads 10 --model  model/ner_gibbs_w2_d2_f4.model  --unigram_model  model/ner_gibbs_w2_d2_f1.model --scoring NER \
--windowL 2 --trainCount 99999 --testCount 99999 --depthL 2 --factorL 4 --verbose false  --train  data/eng_ner/train \
--test data/eng_ner/test --eta 1 --T 4 --c 0 --feat "$feat" --inplace true \
--reward $reward --Q $Q --learning logistic

