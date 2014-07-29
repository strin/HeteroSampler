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
  void runSimple(const Corpus& testCorpus, bool lets_test = true);
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

  /* statistics */
  FeaturePointer tagEntropySimple();
  FeaturePointer wordFrequencies();
  std::pair<Vector2d, std::vector<double> > tagBigram();

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
  ModelTreeUA(const Corpus& corpus, int K);
  ParamPointer gradient(const Sentence& seq);
  double score(const Tag& tag);

  /* parameters */
  double eps, eps_split;

  /* parallel environment */
  virtual void workerThreads(int tid, int seed, std::shared_ptr<MarkovTreeNode>, Tag tag, objcokus rng);
  std::vector<std::shared_ptr<std::thread> > th;
  std::list<std::tuple<int, std::shared_ptr<MarkovTreeNode>, Tag, objcokus> > th_work;
  size_t active_work;
  std::mutex th_mutex;
  std::condition_variable th_cv, th_finished;
  std::vector<std::shared_ptr<std::stringstream> > th_stream;
  std::vector<std::shared_ptr<XMLlog> > th_log;

private:
  void initThreads(size_t numThreads);
};

struct ModelIncrGibbs : public Model {
public:
  ModelIncrGibbs(const Corpus& corpus);
  ParamPointer gradient(const Sentence& seq);
};

struct ModelAdaTree : public ModelTreeUA {
public:
  ModelAdaTree(const Corpus& corpus, int K, double c, double Tstar);
  /* implement components necessary */  
  void workerThreads(int tid, int seed, std::shared_ptr<MarkovTreeNode> node, 
			Tag tag, objcokus rng);
  /* extract posgrad and neggrad for stop-or-not logistic regression */
  std::tuple<double, ParamPointer, ParamPointer, ParamPointer> logisticStop
    (std::shared_ptr<MarkovTreeNode> node, const Sentence& seq, const Tag& tag); 

  FeaturePointer extractStopFeatures
    (std::shared_ptr<MarkovTreeNode> node, const Sentence& seq, const Tag& tag);

  
  double score(std::shared_ptr<MarkovTreeNode> node, const Tag& tag);
private:
  FeaturePointer wordent, wordfreq;
  Vector2d tag_bigram;
  std::vector<double> tag_unigram_start;
  double m_c, m_Tstar;
};
#endif
