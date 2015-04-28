# run Gibbs policy on NER dataset given model pre-trained
./bin/policy --type tagging --policy gibbs --output result/eng_ner/gibbs  --model  model/eng_ner/gibbs.model --train  data/eng_ner/train --test data/eng_ner/test --eta 1 --T 4   --log log/eng_ner/gibbs
