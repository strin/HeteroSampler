#ifndef HETEROSAMPLER_POLICY
#define HETEROSAMPLER_POLICY

#include "utils.h"
#include "model.h"
#include "MarkovTree.h"
#include "tag.h"
#include "ThreadPool.h"
#include <boost/program_options.hpp>

namespace HeteroSampler {

enum MetaFeature { FEAT_BIAS, 
                   FEAT_NB_VARY, 
                   FEAT_NB_ENT, 
                   FEAT_NB_CONSENT, 
                   FEAT_COND_ENT,
                   FEAT_LOG_COND_ENT,
                   FEAT_01_COND_ENT,
                   FEAT_UNIGRAM_ENT,
                   FEAT_INV_UNIGRAM_ENT,
                   FEAT_01_UNIGRAM_ENT,
                   FEAT_LOGISTIC_UNIGRAM_ENT,
                   FEAT_ORACLE,
                   FEAT_COND_LHOOD,
                   FEAT_ORACLE_ENT,
                   FEAT_ORACLE_STALENESS,
                   FEAT_SP,
                   FEAT_LOG_SP,
                   FEAT_EXP_SP,
                   FEAT_SP_COND_ENT,
                   FEAT_SP_UNIGRAM_ENT,
                 };

struct MetaFeatureHash
{
    template <typename T>
    std::size_t operator()(T t) const
    {
        return static_cast<std::size_t>(t);
    }
};

template<class K, class T, class H = std::hash<K>>
class MapInitializer {
public:
  MapInitializer(std::unordered_map<K, T, H>& m)
    :m(m) {
  }

  MapInitializer operator()(const K& key, const T& val) {
    m[key] = val;
    return (*this);
  }

  std::unordered_map<K, T, H>& m;
};

inline static string make_nb(int val, int your_val) {
  return "c-" + tostr(val) + "-" + tostr(your_val);
}

class Policy {
public:
  Policy(ModelPtr model, const boost::program_options::variables_map& vm);
  ~Policy();

  class Result {
    /* object to store the result of applying the policy */
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

    void setNode(size_t i, MarkovTreeNodePtr node) {
      nodes[i] = node;
    }
  };

  typedef std::shared_ptr<Result> ResultPtr;

  struct Action {
  public:
    int pos;
  };

  static ResultPtr makeResultPtr(ptr<Corpus> corpus) {
    return ResultPtr(new Result(corpus));
  }


  /// high-level methods.
  /* apply policy on a new test corpus */
  virtual ResultPtr test(ptr<Corpus> test_corpus);

  /* apply policy on a test corpus with an existing result */
  virtual void test(ResultPtr result);

  /* sub-procedure to test policy */
  virtual void test_policy(ResultPtr result);

  /* train policy on a training corpus */
  virtual void train(ptr<Corpus> corpus);

  /* sub-procedure to train policy */
  virtual void train_policy(ptr<Corpus> corpus);


  /// methods for sampling.
  /* sample node, default uses Gibbs sampling */
  virtual void sample(int tid, MarkovTreeNodePtr node);

  /* wrap model->sampleOne */
  MarkovTreeNodePtr sampleOne(MarkovTreeNodePtr, objcokus& rng, int pos);

  /* update resp of the meta-features */
  void updateResp(MarkovTreeNodePtr node, objcokus& rng, int pos, Heap* heap);

  /* return a number referring to the transition kernel to use.
   * return value:
   *    -1 : stop the markov chain.
   *    a natural number representing a choice.
   */
  virtual Location policy(MarkovTreeNodePtr node) = 0;

  /* estimate delayed reward without making changes to node.
   *   start a rollout of horizon <maxdepth>.
   *      follow the <actions> until the depth > actions.size
   *      then sample action, and push it into <actions>.
   */
  double delayedReward(MarkovTreeNodePtr node, int depth, int maxdepth, vec<int>& actions);

  /* sample delayed reward without making changes to <node> */
  double sampleDelayedReward(MarkovTreeNodePtr node, int id, int maxdepth, int rewardK);

  /* extract meta-features from node */
  virtual FeaturePointer extractFeatures(MarkovTreeNodePtr node, int pos);


  /// dump a node to file.
  /* log information about node */
  virtual void logNode(MarkovTreeNodePtr node);

  /* reset log stream */
  void resetLog(std::shared_ptr<XMLlog> new_lg);

  // response-reward pair.
  struct PolicyExample {  // examples used to train policy.
    double reward;
    double resp;
    double staleness;
    FeaturePointer feat;    // copy and record.
    ParamPointer param;     // copy and record.
    string str, oldstr;     // copy and record.
    MarkovTreeNodePtr node; // just record.
    int choice;
    void serialize(ptr<XMLlog> lg) {
      lg->begin("example");
      lg->logAttr("item", "reward", reward);
      lg->logAttr("item", "staleness", staleness);
      lg->logAttr("item", "resp", resp);
      for (auto& p : *feat) {
        lg->logAttr("feat", p.first, p.second);
      }
      for (auto& p : *param) {
        lg->logAttr("param", p.first, p.second);
      }
      lg->logAttr("item", "choice", choice);
      if (node != nullptr) {
        lg->begin("instance");
        *lg << str << std::endl;
        lg->end();
        lg->begin("old_instance");
        *lg << oldstr << std::endl;
        lg->end();
      }
      lg->end(); // <example>
    }
  };

  bool lets_resp_reward;
  vec<PolicyExample> examples;

  /* const environment. */
  const string name;
  const string learning;
  const int mode_reward, mode_oracle, rewardK;
  const size_t K, Q; // K: num trajectories. Q: num epochs.
  const size_t test_count, train_count;
  const double eta;

  string init_method;

  const bool verbose;
  vec<string> verbose_opt;
  bool verboseOptFind(string verse) {return std::find(verbose_opt.begin(), verbose_opt.end(), verse) != verbose_opt.end(); }

  const bool lets_inplace;              // not work with entire history.
  // feature option, each string switches a meta-feature to add.

  vec<MetaFeature> feat;
  map<string, MetaFeature> feat_map;
  std::unordered_map<MetaFeature, string, MetaFeatureHash> feat_name;

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
  //       second/third pass only update words with entropy exceeding threshold.
  virtual Location policy(MarkovTreeNodePtr node);

  size_t T; // how many sweeps.
};

class BlockPolicy : public Policy {
public:
  BlockPolicy(ModelPtr model, const variables_map& vm);

  ~BlockPolicy();

  class Result : public Policy::Result {
  public:
    Result(ptr<Corpus> corpus);
    Heap heap;
  };

  typedef ptr<BlockPolicy::Result> ResultPtr;

  virtual MarkovTreeNodePtr sampleOne(ResultPtr result,
                                      objcokus& rng,
                                      const Location& loc);

  /* overrides inherited virtual method,
   * calls method test(corpus, budget = 1) */
  virtual Policy::ResultPtr test(ptr<Corpus> corpus);
  virtual ResultPtr test(ptr<Corpus> corpus, double budget);

  /* overrides inherited virtual method,
   * calls method test(result, budget = 1) */
  virtual void test(Policy::ResultPtr result);
  virtual void test(ResultPtr result, double budget);

  /* overrides inherited virtual method,
   * calls method test_policy(result, budget = 1) */
  virtual void test_policy(Policy::ResultPtr result);
  virtual void test_policy(ResultPtr result, double budget);

  virtual Location policy(MarkovTreeNodePtr node);
  virtual Location policy(ResultPtr result);
};

}

#endif
