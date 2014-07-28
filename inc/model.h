#ifndef POS_MODEL_H
#define POS_MODEL_H

#include "tag.h"
#include "corpus.h"
#include "objcokus.h"
#include "log.h"
#include "MarkovTree.h"

#include <vector>
#include <list>
#include <thread>
#include <condition_variable>

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
  int T, B, Q, Q0;
  double testFrequency;
  double eta;
  std::vector<objcokus> rngs;
protected:
  /* parameters */
  int K;
};


struct ModelTreeUA : public Model {
public:
  ModelTreeUA(const Corpus& corpus);
  ParamPointer gradient(const Sentence& seq);
  double score(const Tag& tag);

  /* parameters */
  double eps, eps_split;

  /* parallel environment */
  void workerThreads(int id, std::shared_ptr<MarkovTreeNode>, Tag tag, objcokus cokus);
  std::vector<std::shared_ptr<std::thread> > th;
  std::list<std::tuple<int, std::shared_ptr<MarkovTreeNode>, Tag, objcokus> > th_work;
  size_t active_work;
  std::mutex th_mutex;
  std::condition_variable th_cv, th_finished;

private:
  void initThreads(size_t numThreads);
};

struct ModelIncrGibbs : public Model {
public:
  ModelIncrGibbs(const Corpus& corpus);
  ParamPointer gradient(const Sentence& seq);
};
#endif
