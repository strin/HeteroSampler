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
  void runSimple(const Corpus& testCorpus);
  void run(const Corpus& testCorpus);
  double test(const Corpus& corpus);

  /* gradient interface */
  virtual ParamPointer gradient(const Sentence& seq); 
  void adagrad(ParamPointer gradient);

  /* default implementation */
  ParamPointer gradientGibbs(const Sentence& seq);
  ParamPointer gradientSimple(const Sentence& seq);

  const Corpus& corpus;
  XMLlog xmllog;
  ParamPointer param, G2;

  /* parameters */
  int T, B, K, Q, Q0;
  double testFrequency;
  double eta;
  std::vector<objcokus> rngs;
};

struct ModelTreeUA : public Model {
public:
  ModelTreeUA(const Corpus& corpus);
  ParamPointer gradient(const Sentence& seq);
  double score(const Tag& tag);

  /* parameters */
  double eps, eps_split;
};

struct ModelIncrGibbs : public Model {
public:
  ModelIncrGibbs(const Corpus& corpus);
  ParamPointer gradient(const Sentence& seq);
};
#endif
