// Train Stopping Rules based on Statistics of Markov chains. 
#ifndef POS_STOP_H
#define POS_STOP_H

#include "utils.h"
#include "model.h"
#include "MarkovTree.h"
#include "tag.h"
#include "ThreadPool.h"
#include <boost/program_options.hpp>

// use logistic regression to predict stop or not.
class Stop {
public:
  // constructor.
  // model: the Markov chain that generates samples. 
  // B    : the number of first steps used to generate StopDataset.
  // T    : the overall number of steps. 
  Stop(ModelPtr model, const boost::program_options::variables_map& vm);
  ~Stop();

  // explore example (Sentence). 
  StopDatasetPtr explore(const Sentence& seq);

  // sample the Markov chain, and gather reward information.
  void sample(int tid, MarkovTreeNodePtr node);

  // sapmle test data using Markov chain with optimal stopping.
  void sampleTest(int tid, MarkovTreeNodePtr node);

  // extract features for logistic regression.
  FeaturePointer extractStopFeatures(MarkovTreeNodePtr node);
  double score(MarkovTreeNodePtr node);

  // run the experiment.
  void run(const Corpus& corpus);

  // test stop or not.
  double test(const Corpus& testCorpus);

private:
  // const environment. 
  FeaturePointer wordent, wordfreq;
  Vector2d tag_bigram;
  std::vector<double> tag_unigram_start;
  size_t train_count, test_count;
  std::shared_ptr<XMLlog> lg;
  const bool lets_adaptive;
  const size_t iter;

  // global environment.
  ModelPtr model;
  objcokus rng;
  size_t T, B, K;
  double eta, c;
  std::string name;
  StopDatasetPtr stop_data;
  std::shared_ptr<XMLlog> stop_data_log;
  ParamPointer param, G2;
  FeaturePointer mean_feat, std_feat;

  // parallel environment.
  ThreadPool<MarkovTreeNodePtr> thread_pool, test_thread_pool;
};

#endif
