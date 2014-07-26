#include "corpus.h"
#include "objcokus.h"
#include "tag.h"
#include "model.h"
#include <iostream>

using namespace std;

int main() {
  Corpus corpus;
  corpus.read("data/eng_ner/train");
  Corpus testCorpus;
  testCorpus.read("data/eng_ner/test");
  Model model(corpus);
  model.run(testCorpus);
}
