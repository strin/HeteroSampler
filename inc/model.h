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

#include <boost/program_options.hpp>


inline static void adagrad(ParamPointer param, ParamPointer G2, ParamPointer gradient, double eta) {
  for(const std::pair<std::string, double>& p : *gradient) {
    mapUpdate(*G2, p.first, p.second * p.second);
    mapUpdate(*param, p.first, eta * p.second/sqrt(1e-4 + (*G2)[p.first]));
  }
}

struct Model {
public:
  Model(const Corpus* corpus, const boost::program_options::variables_map& vm);
  virtual void run(const Corpus& testCorpus, bool lets_test = true);
  double test(const Corpus& testCorpus);

  /* gradient interface */
  virtual ParamPointer gradient(const Sentence& seq) = 0; 
  virtual TagVector sample(const Sentence& seq) = 0;
  // infer under computational constraints.
  // emulate t-step transition of a markov chain.
  // default: use Model::sample(*tag.seq), i.e. time = T.
  virtual void sample(Tag& tag, int time);             // inplace.
  virtual void sampleOne(Tag& tag, int choice);           // sample using custom kernel choice.

  /* stats utils */
  std::tuple<ParamPointer, double> tagEntropySimple() const;
  std::tuple<ParamPointer, double> wordFrequencies() const;
  std::pair<Vector2d, std::vector<double> > tagBigram() const;
  virtual double score(const Tag& tag);
  // evaluate the accuracy for POS tag aginst truth.
  // return 0: hit count.
  // return 1: pred count.
  std::tuple<int, int> evalPOS(const Tag& tag);
  // evaulate the F1 score for NER tag aginst truth.
  // return 0: hit count.
  // return 1: pred count.
  // return 2: truth count.
  std::tuple<int, int, int> evalNER(const Tag& tag);

  /* parameters */
  size_t T, B, Q;
  int Q0;
  double testFrequency;
  double eta;
  std::vector<objcokus> rngs;
  const Corpus* corpus;
  ParamPointer param, G2, stepsize;   // model.

  /* IO */
  friend std::ostream& operator<<(std::ostream& os, const Model& model);
  friend std::istream& operator>>(std::istream& os, Model& model);

  XMLlog xmllog;
  virtual void logArgs(); 
  /* const environment */
  enum Scoring {SCORING_NER, SCORING_ACCURACY };
  Scoring scoring;
  const boost::program_options::variables_map& vm;

protected:
  void adagrad(ParamPointer gradient);
  void configStepsize(FeaturePointer gradient, double new_eta);

  int K;          // num of particle. 
  int num_ob;     // current number of observations.
};

typedef std::shared_ptr<Model> ModelPtr;

struct ModelSimple : public Model {
public:
  ModelSimple(const Corpus* corpus, const boost::program_options::variables_map& vm);
  ParamPointer gradient(const Sentence& seq, TagVector* vec = nullptr, bool update_grad = true);
  ParamPointer gradient(const Sentence& seq);
  TagVector sample(const Sentence& seq); 
  void sample(Tag& tag, int time);
  FeaturePointer extractFeatures(const Tag& tag, int pos);
  
  virtual void logArgs();
protected:
  int windowL, depthL; // range of unigram features.
};

struct ModelCRFGibbs : public ModelSimple {
public:
  ModelCRFGibbs(const Corpus* corpus, const boost::program_options::variables_map& vm);
  ParamPointer gradient(const Sentence& seq, TagVector* vec = nullptr, bool update_grad = true);
  ParamPointer gradient(const Sentence& seq);
  TagVector sample(const Sentence& seq);
  void sample(Tag& tag, int time);
  void sampleOne(Tag& tag, int choice);
  void addUnigramFeatures(const Tag& tag, int pos, FeaturePointer features);
  void addBigramFeatures(const Tag& tag, int pos, FeaturePointer features);
  std::function<FeaturePointer(const Tag& tag, int pos)> extractFeatures;
  // FeaturePointer extractFeatures(const Tag& tag, int pos);
  FeaturePointer extractFeaturesAll(const Tag& tag);
  double score(const Tag& tag);

  virtual void logArgs();
private:
  void sampleOneSweep(Tag& tag);
  int factorL;
};

struct ModelIncrGibbs : public ModelCRFGibbs {
public:
  ModelIncrGibbs(const Corpus* corpus, const boost::program_options::variables_map& vm);
  ParamPointer gradient(const Sentence& seq, TagVector* vec = nullptr, bool update_grad = true);
  ParamPointer gradient(const Sentence& seq);
  TagVector sample(const Sentence& seq);
};


struct ModelTreeUA : public ModelCRFGibbs {
public:
  ModelTreeUA(const Corpus* corpus, const boost::program_options::variables_map& vm);

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
  ModelAdaTree(const Corpus* corpus, const boost::program_options::variables_map& vm);
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
  ParamPointer wordent, wordfreq;
  double wordent_mean, wordfreq_mean;
  Vector2d tag_bigram;
  std::vector<double> tag_unigram_start;
  double m_c, m_Tstar;
};

#endif
