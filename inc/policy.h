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

#define POLICY_MARKOV_CHAIN_MAXDEPTH 10000000

namespace Tagging {
  const static string NB_VARY = "nb-#vary";
  const static string NER_DISAGREE = "ner-disagree";
  const static string NER_DISAGREE_L = "ner-disagree-l";
  const static string NER_DISAGREE_R = "ner-disagree-r";

  class Policy {
  public:
    Policy(ModelPtr model, const boost::program_options::variables_map& vm);
    ~Policy();

    // results in terms of test samples.
    class Result {
    public:
      Result(ptr<Corpus> corpus);
      std::vector<MarkovTreeNodePtr> nodes;
      ptr<Corpus> corpus;
      double score;
      double time;
      double wallclock;
      double wallclock_policy, wallclock_sample;

      size_t size() const {
        return nodes.size();
      }
      MarkovTreeNodePtr getNode(size_t i) const {
        return nodes[i];
      }
    };
    struct ROC {
    public:
      ROC() : TP(0), FP(0), TN(0), FN(0), threshold(0) {}
      double TP, FP, TN, FN;
      double threshold;
      double prec_sample, prec_stop, recall_sample, recall_stop;
      string str() const {
      	string res = "";
      	res += "threshold (" + std::to_string(threshold) + ")\t";
      	res += "prec/sample (" + std::to_string(prec_sample) + ")\t";
      	res += "recall/sample (" + std::to_string(recall_sample) + ")\t";
      	res += "prec/stop (" + std::to_string(prec_stop) + ")\t";
      	res += "recall/stop (" + std::to_string(recall_stop) + ")\t";
      	return res;
      }
    };

    typedef std::shared_ptr<Result> ResultPtr;
    inline static ResultPtr makeResultPtr(ptr<Corpus> corpus) {
      return ResultPtr(new Result(corpus));
    }

    // run test on corpus.
    // return: accuracy on the test set.
    ResultPtr test(ptr<Corpus> testCorpus);
    void test(ResultPtr result);
    virtual void testPolicy(ResultPtr result);

    // apply gradient from samples to policy.
    virtual void gradientPolicy(MarkovTree& tree);

    // apply gradient from samples to model.
    virtual void gradientKernel(MarkovTree& tree);

    // run training on corpus.
    virtual void train(ptr<Corpus> corpus);

    // train policy.
    virtual void trainPolicy(ptr<Corpus> corpus);

    // train primitive kernels. 
    virtual void trainKernel(ptr<Corpus> corpus);

    // sample node, default uses Gibbs sampling.
    virtual void sampleTest(int tid, MarkovTreeNodePtr node);

    // sample node, for training. default: call sampleTest.
    virtual void sample(int tid, MarkovTreeNodePtr node);
    
    // wrap model->sampleOne.
    MarkovTreeNodePtr sampleOne(MarkovTreeNodePtr, objcokus& rng, int pos);

    // update resp of the meta-features.
    void updateResp(MarkovTreeNodePtr node, objcokus& rng, int pos, Heap* heap);

    // return a number referring to the transition kernel to use.
    // return = -1 : stop the markov chain.
    // o.w. return a natural number representing a choice.
    virtual int policy(MarkovTreeNodePtr node) = 0;

    // estimate the reward of a MarkovTree node (default: -dist).
    virtual double reward(MarkovTreeNodePtr node);

    // extract features from node.
    virtual FeaturePointer extractFeatures(MarkovTreeNodePtr node, int pos);

    // compute the checksum of a node. (determine if the feature needs to be re-extracted due to change in neighbors).
    virtual double checksum(MarkovTreeNodePtr node, int pos);

    // log information about node.
    virtual void logNode(MarkovTreeNodePtr node);

    // reset log.
    void resetLog(std::shared_ptr<XMLlog> new_lg);

    // response-reward pair.
    bool lets_resp_reward;
    vec<pair<double, double> > resp_RL, test_resp_RL; // incr in correctness, lower bound of R. 
    vec<pair<double, double> > resp_RH, test_resp_RH; // whether incorrect, upper bound of R.
    vec<pair<double, double> > resp_reward, test_resp_reward; // true reward.
    vec<tuple<double, double, string, int> > test_word_tag;                //  corresponding word, tag pair.
      
    // compute TP, FP, TN, FN.
    vec<ROC> getROC(const int fold[], const int num_fold, std::vector<std::pair<double, double> >& resp_reward);

    /* const environment. */
    ParamPointer wordent, wordfreq;
    double wordent_mean, wordfreq_mean;
    Vector2d tag_bigram;
    std::vector<double> tag_unigram_start;
    const std::string name;
    const size_t K, Q; // K: num samples. Q: num epochs.
    const size_t test_count, train_count;
    const double eta;

    string init_method;

    const bool verbose; 
    vec<string> verbose_opt;
    bool verboseOptFind(string verse) {return std::find(verbose_opt.begin(), verbose_opt.end(), verse) != verbose_opt.end(); }

    const bool lets_inplace;              // not work with entire history.
    const bool lets_lazymax;              // take max sample only after each sweep. 
    int lazymax_lag;        // the lag to take max, default(-1, entire instance).
    // feature option, each string switches a meta-feature to add.

    vec<string> featopt; 
    bool featoptFind(string feat) {return std::find(featopt.begin(), featopt.end(), feat) != featopt.end(); }

    /* global environment. */
    ModelPtr model;                 // full model.
    ModelPtr model_unigram;         // unigram/lower-order model.    

    objcokus rng;
    std::shared_ptr<XMLlog> lg;
    ParamPointer param, G2;

    /* parallel environment. */
    ThreadPool<MarkovTreeNodePtr> thread_pool, test_thread_pool;    
  };


  // baseline policy of selecting Gibbs sampling kernels 
  // just based on Gibbs sweeping.
  class GibbsPolicy : public Policy {
  public:
    GibbsPolicy(ModelPtr model, const boost::program_options::variables_map& vm);

    // policy: first make an entire pass over the sequence. 
    //	     second/third pass only update words with entropy exceeding threshold.
    int policy(MarkovTreeNodePtr node);

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

  // cyclic policy, 
  // first sweep samples everything. 
  // subsequent sweeps use logistic regression to predict whether to sample. 
  // at end of every sweep, predict stop or not.
  class CyclicPolicy : public Policy {
  public:
    CyclicPolicy(ModelPtr model, const boost::program_options::variables_map& vm);

    virtual int policy(MarkovTreeNodePtr node); 
    
    // reward = -dist - c * (depth+1).
    double reward(MarkovTreeNodePtr node);
    
    // extract features from node.
    virtual FeaturePointer extractFeatures(MarkovTreeNodePtr node, int pos);

    double c;          // regularization of computation.
  };

  // policy based on learning value function.
  // virtual class that overwrites the training function.
  class CyclicValuePolicy : public CyclicPolicy {
  public:
    CyclicValuePolicy(ModelPtr model, const boost::program_options::variables_map& vm);

    virtual int policy(MarkovTreeNodePtr node);

    // sample for training.
    virtual void sample(int tid, MarkovTreeNodePtr node);

    // training.
    virtual void trainPolicy(ptr<Corpus> corpus);

    // testing.
    virtual void testPolicy(Policy::ResultPtr result);

  };

  class MultiCyclicValuePolicy : public CyclicValuePolicy {
  public:
    MultiCyclicValuePolicy(ModelPtr model, const boost::program_options::variables_map& vm);

    virtual int policy(MarkovTreeNodePtr node);

    // logNodes after each pass.
    virtual void logNode(MarkovTreeNodePtr node);

    virtual void sample(int tid, MarkovTreeNodePtr node);
    
    // rename features from CyclicPolicy::extractFeatures based on current pass.
    virtual FeaturePointer extractFeatures(MarkovTreeNodePtr node, int pos);

  protected:
    size_t T;
  };

  class MultiCyclicValueUnigramPolicy : public MultiCyclicValuePolicy {
  public:
    MultiCyclicValueUnigramPolicy(ModelPtr model, ModelPtr model_unigram, const boost::program_options::variables_map& vm);

    // add features inpired by the unigram.
    virtual FeaturePointer extractFeatures(MarkovTreeNodePtr node, int pos);

  protected:
    ModelPtr model_unigram;
  };

  // Baseline Random Scan Gibbs Sampler.
  // first take a cyclic sweep, then sample uniform at random.
  class RandomScanPolicy : public Policy {
  public:
    RandomScanPolicy(ModelPtr model, const boost::program_options::variables_map& vm);
    virtual int policy(MarkovTreeNodePtr node);
    virtual void sample(int tid, MarkovTreeNodePtr node);
    virtual FeaturePointer extractFeatures(MarkovTreeNodePtr node, int pos);
  protected:
    double Tstar;
    int windowL;
  };

  // Lock-down sampler.
  // lock-down position above the threshold and sample. 
  // when meta-features are unigram, essentially same as multi-cyclic scan.
  // but stop is more natural (all positions below thrsehold). 
  class LockdownPolicy : public Policy {
  public:
    LockdownPolicy(ModelPtr model, const boost::program_options::variables_map& vm);
    virtual int policy(MarkovTreeNodePtr node);
    FeaturePointer extractFeatures(MarkovTreeNodePtr node, int pos);
    void sample(int tid, MarkovTreeNodePtr node);

    double c;
    size_t T;
  };
}
#endif
