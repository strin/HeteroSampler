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

typedef std::shared_ptr<std::vector<std::string> > StringVector;
static StringVector makeStringVector() {
  return StringVector(new std::vector<std::string>());
}

struct Model {
public:
  Model(const Corpus* corpus, int T = 1, int B = 0, int Q = 10, double eta = 0.5);
  virtual void run(const Corpus& testCorpus, bool lets_test = true);
  double test(const Corpus& testCorpus, int time);

  /* gradient interface */
  virtual ParamPointer gradient(const Sentence& seq) = 0; 
  virtual TagVector sample(const Sentence& seq) = 0;
  // infer under computational constraints.
  // emulate t-step transition of a markov chain.
  // default: use Model::sample(*tag.seq), i.e. time = T.
  virtual void sample(Tag& tag, int time);             // inplace.

  /* stats utils */
  FeaturePointer tagEntropySimple() const;
  FeaturePointer wordFrequencies() const;
  std::pair<Vector2d, std::vector<double> > tagBigram() const;
  static StringVector NLPfunc(const std::string word);

  /* parameters */
  int T, B, Q, Q0;
  double testFrequency;
  double eta;
  std::vector<objcokus> rngs;
  const Corpus* corpus;
  ParamPointer param, G2, stepsize;   // model.

  /* IO */
  friend std::ostream& operator<<(std::ostream& os, const Model& model);
  friend std::istream& operator>>(std::istream& os, Model& model);

  XMLlog xmllog;
protected:
  void adagrad(ParamPointer gradient);
  void configStepsize(ParamPointer gradient, double new_eta);

  int K;          // num of particle. 
  int num_ob;     // current number of observations.

  static std::unordered_map<std::string, StringVector> word_feat;
};

typedef std::shared_ptr<Model> ModelPtr;

struct ModelSimple : public Model {
public:
  ModelSimple(const Corpus* corpus, int windowL = 0, int T = 1, int B = 0, int Q = 10, double eta = 0.5);
  ParamPointer gradient(const Sentence& seq, TagVector* vec = nullptr, bool update_grad = true);
  ParamPointer gradient(const Sentence& seq);
  TagVector sample(const Sentence& seq); 
  void sample(Tag& tag, int time);
  FeaturePointer extractFeatures(const Tag& tag, int pos);

protected:
  int windowL;
};

struct ModelCRFGibbs : public ModelSimple {
public:
  ModelCRFGibbs(const Corpus* corpus, int windowL = 0, int T = 1, int B = 0, int Q = 10, double eta = 0.5);
  ParamPointer gradient(const Sentence& seq, TagVector* vec = nullptr, bool update_grad = true);
  ParamPointer gradient(const Sentence& seq);
  TagVector sample(const Sentence& seq);
  void sample(Tag& tag, int time);
  void addUnigramFeatures(const Tag& tag, int pos, FeaturePointer features);
  void addBigramFeatures(const Tag& tag, int pos, FeaturePointer features);
  FeaturePointer extractFeatures(const Tag& tag, int pos);
  FeaturePointer extractFeatures(const Tag& tag);

private:
  void sampleOneSweep(Tag& tag);
};

struct ModelIncrGibbs : public ModelCRFGibbs {
public:
  ModelIncrGibbs(const Corpus* corpus, int windowL = 0, int T = 1, int B = 0, int Q = 10, double eta = 0.5);
  ParamPointer gradient(const Sentence& seq, TagVector* vec = nullptr, bool update_grad = true);
  ParamPointer gradient(const Sentence& seq);
  TagVector sample(const Sentence& seq);
};

/* using forward-backward inference for discriminative MRF */
struct ModelFwBw : public ModelCRFGibbs {
public:
  ModelFwBw(const Corpus* corpus, int windowL = 0, int T = 1, int B = 0, int Q = 10, double eta = 0.5);
  ParamPointer gradient(const Sentence& seq, TagVector* samples = nullptr, bool update_grad = true);
  ParamPointer gradient(const Sentence& seq);
  TagVector sample(const Sentence& seq);
};

struct ModelTreeUA : public ModelCRFGibbs {
public:
  ModelTreeUA(const Corpus* corpus, int windowL = 0, int K = 5, 
	      int T = 1, int B = 0, int Q = 10, int Q0 = 1, double eta = 0.5);

  virtual void run(const Corpus& testCorpus);

  virtual std::shared_ptr<MarkovTree> explore(const Sentence& seq);
  virtual ParamPointer gradient(const Sentence& seq);
  virtual TagVector sample(const Sentence& seq);
  virtual double score(const Tag& tag);

  /* parameters */
  double eps, eps_split;

  /* parallel environment */
  virtual void workerThreads(int tid, std::shared_ptr<MarkovTreeNode> node, Tag tag);
  std::vector<std::shared_ptr<std::thread> > th;
  std::list<std::tuple<std::shared_ptr<MarkovTreeNode>, Tag> > th_work;
  size_t active_work;
  std::mutex th_mutex;
  std::condition_variable th_cv, th_finished;
  std::vector<std::shared_ptr<std::stringstream> > th_stream;
  std::vector<std::shared_ptr<XMLlog> > th_log;

protected:
  int Q0;
  void initThreads(size_t numThreads);
};


struct ModelAdaTree : public ModelTreeUA {
public:
  ModelAdaTree(const Corpus* corpus, int windowL = 0, int K = 5, 
	      double c = 1, double Tstar = 0, double etaT = 0.05, 
	      int T = 1, int B = 0, int Q = 10, int Q0 = 1, double eta = 0.5);
  /* implement components necessary */  
  void workerThreads(int tid, MarkovTreeNodePtr node, Tag tag);
  /* extract posgrad and neggrad for stop-or-not logistic regression */
  std::tuple<double, ParamPointer, ParamPointer, FeaturePointer> logisticStop
    (MarkovTreeNodePtr node, const Sentence& seq, const Tag& tag); 

  /* stop feature extraction for each word */
  virtual FeaturePointer extractStopFeatures
    (MarkovTreeNodePtr node, const Sentence& seq, const Tag& tag, int pos);
  /* stop feature extraction for entire sentence */
  virtual FeaturePointer extractStopFeatures
    (MarkovTreeNodePtr node, const Sentence& seq, const Tag& tag);

  double score(MarkovTreeNodePtr node, const Tag& tag);

  /* parameters */
  double etaT;
private:
  FeaturePointer wordent, wordfreq;
  Vector2d tag_bigram;
  std::vector<double> tag_unigram_start;
  double m_c, m_Tstar;
};

struct ModelPrune : public ModelAdaTree {
public:
  ModelPrune(const Corpus* corpus, int windowL = 0, int K = 5, int prune_mode = 1,  
	      size_t data_size = 100, double c = 1, double Tstar = 0, double etaT = 0.05, 
	      int T = 1, int B = 0, int Q = 10, int Q0 = 1, double eta = 0.5);

  void workerThreads(int td, std::shared_ptr<MarkovTreeNode> node, Tag tag);
  std::shared_ptr<MarkovTree> explore(const Sentence& seq);
protected:
  StopDatasetPtr stop_data;
  std::shared_ptr<XMLlog> stop_data_log;
  size_t data_size;
  int prune_mode;
};

struct ModelPruneInd : public ModelPrune {
public: 
  ModelPruneInd(const Corpus* corpus, int windowL = 0, int K = 5, int prune_mode = 1,  
	      size_t data_size = 100, double c = 1, double Tstar = 0, double etaT = 0.05, 
	      int T = 1, int B = 0, int Q = 10, int Q0 = 1, double eta = 0.5);
  
  void workerThreads(int td, std::shared_ptr<MarkovTreeNode> node, Tag tag);
  std::shared_ptr<MarkovTree> explore(const Sentence& seq);
};
#endif
