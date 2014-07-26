#ifndef POS_MODEL_H
#define POS_MODEL_H

#include "tag.h"
#include "corpus.h"
#include "objcokus.h"
#include "log.h"

#include <vector>

struct Model {
public:
  Model(const Corpus& corpus);
  void run(const Corpus& testCorpus);
  double test(const Corpus& corpus);

  ParamPointer gradientGibbs(const Sentence& seq);
  void adagrad(ParamPointer gradient);

  const Corpus& corpus;
  XMLlog log;
  ParamPointer param, G2;

  /* parameters */
  int T, B, numThreads, Q;
  double testFrequency;
  double eta;
  std::vector<objcokus> rngs;
};

#endif
