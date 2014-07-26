#ifndef POS_MODEL_H
#define POS_MODEL_H

#include "tag.h"
#include "corpus.h"
#include "objcokus.h"
#include <vector>

struct Model {
public:
  Model(const Corpus& corpus);
  void run(const Corpus& testCorpus);
  double test(const Corpus& corpus);

  ParamPointer gradientGibbs(const Sentence& seq);
  void adagrad(ParamPointer gradient);

  const Corpus& corpus;
  ParamPointer param, G2;

  /* parameters */
  int T = 2, B = 0, numThreads = 4, Q = 10;
  double testFrequency = 0.1;
  double eta = 0.5;
  std::vector<objcokus> rngs;
};

#endif
