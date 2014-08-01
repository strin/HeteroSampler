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
  Model(const Corpus& corpus, int T = 1, int B = 0, int Q = 10, double eta = 0.5);
  virtual void run(const Corpus& testCorpus, bool lets_test = true);
  double test(const Corpus& testCorpus);

  /* gradient interface */
  virtual ParamPointer gradient(const Sentence& seq) = 0; 
  virtual TagVector sample(const Sentence& seq) = 0;

  /* stats utils */
  FeaturePointer tagEntropySimple() const;
  FeaturePointer wordFrequencies() const;
  std::pair<Vector2d, std::vector<double> > tagBigram() const;
  static std::vector<std::string> NLPfunc(const std::string word);

  /* parameters */
  int T, B, Q, Q0;
  double testFrequency;
  double eta;
  std::vector<objcokus> rngs;
  const Corpus& corpus;
  ParamPointer param, G2, stepsize;   // model.

protected:
  void adagrad(ParamPointer gradient);
  void configStepsize(ParamPointer gradient, double new_eta);

  int K;          // num of particle. 

  XMLlog xmllog;
};

struct ModelSimple : public Model {
public:
  ModelSimple(const Corpus& corpus, int windowL = 0, int T = 1, int B = 0, int Q = 10, double eta = 0.5);
  ParamPointer gradient(const Sentence& seq, TagVector* vec = nullptr, bool update_grad = true);
  ParamPointer gradient(const Sentence& seq);
  TagVector sample(const Sentence& seq); 
  FeaturePointer extractFeatures(const Tag& tag, int pos);

protected:
  int windowL;
};

struct ModelCRFGibbs : public ModelSimple {
public:
  ModelCRFGibbs(const Corpus& corpus, int windowL = 0, int T = 1, int B = 0, int Q = 10, double eta = 0.5);
  ParamPointer gradient(const Sentence& seq, TagVector* vec = nullptr, bool update_grad = true);
  ParamPointer gradient(const Sentence& seq);
  TagVector sample(const Sentence& seq);
  FeaturePointer extractFeatures(const Tag& tag, int pos);
  FeaturePointer extractFeatures(const Tag& tag);
};

struct ModelIncrGibbs : public ModelCRFGibbs {
public:
  ModelIncrGibbs(const Corpus& corpus, int windowL = 0, int T = 1, int B = 0, int Q = 10, double eta = 0.5);
  ParamPointer gradient(const Sentence& seq, TagVector* vec = nullptr, bool update_grad = true);
  ParamPointer gradient(const Sentence& seq);
  TagVector sample(const Sentence& seq);
};

/* using forward-backward inference for discriminative MRF */
struct ModelFwBw : public ModelCRFGibbs {
public:
  ModelFwBw(const Corpus& corpus, int windowL = 0, int T = 1, int B = 0, int Q = 10, double eta = 0.5);
  ParamPointer gradient(const Sentence& seq, TagVector* vec = nullptr, bool update_grad = true);
  ParamPointer gradient(const Sentence& seq);
  TagVector sample(const Sentence& seq);
};

struct ModelTreeUA : public ModelCRFGibbs {
public:
  ModelTreeUA(const Corpus& corpus, int windowL = 0, int K = 5, int T = 1, int B = 0, int Q = 10, double eta = 0.5);

  void run(const Corpus& testCorpus);

  std::shared_ptr<MarkovTree> explore(const Sentence& seq);
  ParamPointer gradient(const Sentence& seq);
  TagVector sample(const Sentence& seq);

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


struct ModelAdaTree : public ModelTreeUA {
public:
  ModelAdaTree(const Corpus& corpus, int windowL = 0, int K = 5, double c = 1, double Tstar = 10, 
	      double etaT = 0.5, int T = 1, int B = 0, int Q = 10, double eta = 0.5);
  /* implement components necessary */  
  void workerThreads(int tid, int seed, std::shared_ptr<MarkovTreeNode> node, 
			Tag tag, objcokus rng);
  /* extract posgrad and neggrad for stop-or-not logistic regression */
  std::tuple<double, ParamPointer, ParamPointer, ParamPointer> logisticStop
    (std::shared_ptr<MarkovTreeNode> node, const Sentence& seq, const Tag& tag); 

  FeaturePointer extractStopFeatures
    (std::shared_ptr<MarkovTreeNode> node, const Sentence& seq, const Tag& tag);

  double score(std::shared_ptr<MarkovTreeNode> node, const Tag& tag);

  /* parameters */
  double etaT;
private:
  FeaturePointer wordent, wordfreq;
  Vector2d tag_bigram;
  std::vector<double> tag_unigram_start;
  double m_c, m_Tstar;
};
#endif
