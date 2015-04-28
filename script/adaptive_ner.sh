# run adaptive policy on NER 
./bin/policy --type tagging --policy adaptive --output result/eng_ner/adaptive  --model  model/eng_ner/gibbs.model --train  data/eng_ner/train --test data/eng_ner/test --eta 1 --T 6  --feat 'sp cond-ent bias nb-vary nb-discord' --reward 0  --log log/eng_ner/adaptive
