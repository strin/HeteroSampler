// Learning inference policies. 
//
#ifndef POS_POLICY_H
#define POS_POLICY_H

#include "utils.h"
#include "model.h"
#include "MarkovTree.h"
#include "tag.h"
#include "ThreadPool.h"
#include <boost/program_options.hpp>

#define POLICY_MARKOV_CHAIN_MAXDEPTH 10000

class Policy {
public:
  Policy(ModelPtr model, const boost::program_options::variables_map& vm);
  ~Policy();

  // run test on corpus.
  // return: accuracy on the test set.
  double test(const Corpus& corpus);

  // sample node, default uses Gibbs sampling.
  void sampleTest(int tid, MarkovTreeNodePtr node);

  // return a number referring to the transition kernel to use.
  // return = -1 : stop the markov chain.
  // o.w. return a natural number representing a choice.
  virtual int policy(MarkovTreeNodePtr node) = 0;

  // extract features from node.
  virtual FeaturePointer extractFeatures(MarkovTreeNodePtr node, int pos);

  // log information about node.
  virtual void logNode(MarkovTreeNodePtr node);
protected:
  // const environment. 
  FeaturePointer wordent, wordfreq;
  Vector2d tag_bigram;
  std::vector<double> tag_unigram_start;
  const std::string name;
  const size_t test_count, train_count;

  // global environment.
  ModelPtr model;
  objcokus rng;
  std::shared_ptr<XMLlog> lg;

  // parallel environment.
  ThreadPool<MarkovTreeNodePtr> test_thread_pool;    
};


// baseline policy of selecting Gibbs sampling kernels 
// just based on Gibbs sweeping.
class GibbsPolicy : public Policy {
public:
  GibbsPolicy(ModelPtr model, const boost::program_options::variables_map& vm);

  // policy: first make an entire pass over the sequence. 
  //	     second/third pass only update words with entropy exceeding threshold.
  int policy(MarkovTreeNodePtr node);


private:
  size_t T; // how many sweeps.
};

// baseline policy of selecting Gibbs sampling kernels 
// just based on thresholding entropy of tags for words.
class EntropyPolicy : public Policy   {
public:
  EntropyPolicy(ModelPtr model, const boost::program_options::variables_map& vm);

  // policy: first make an entire pass over the sequence. 
  //	     second/third pass only update words with entropy exceeding threshold.
  int policy(MarkovTreeNodePtr node);


private:
  double threshold;   // entropy threshold = log(threshold).
};
#endif
