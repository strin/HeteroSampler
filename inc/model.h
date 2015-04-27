#ifndef POS_MODEL_H
#define POS_MODEL_H

#include "tag.h"
#include "gm.h"
#include "corpus.h"
#include "MarkovTree.h"

namespace HeteroSampler {
  inline static void adagrad(ParamPointer param, ParamPointer G2, ParamPointer gradient, double eta) {
    for(const std::pair<std::string, double>& p : *gradient) {
      mapUpdate(*G2, p.first, p.second * p.second);
      mapUpdate(*param, p.first, eta * p.second/sqrt(1e-4 + (*G2)[p.first]));
    }
  }

  struct Model {
  public:
    Model(ptr<Corpus> corpus, const boost::program_options::variables_map& vm);
    virtual void run(ptr<Corpus> test_corpus, bool lets_test = true);
    double test(ptr<Corpus> test_corpus);

    /* gradient interface */
    virtual ParamPointer gradient(const Instance& seq) = 0;
    virtual TagVector sample(const Instance& seq, bool argmax = false) = 0;
    // infer under computational constraints.
    // emulate t-step transition of a markov chain.
    // default: use Model::sample(*tag.seq), i.e. time = T.
    virtual void sample(Tag& tag, int time, bool argmax = false);             // inplace.

    // sample using custom kernel choice.
    // use Gibbs to sample <choice> with random number generator <rng> and feature extraction functional <feat_extract>
    // flags:
    //  use_meta_feature: only reward, tags, and sc would be updated if set true.
    virtual void sampleOne(GraphicalModel& gm, objcokus& rng, int choice, bool use_meta_feature = true);

    // sample using custom kernel choice at initialization.
    // only applies if "init" flag is on (not equal to *random*).
    virtual void sampleOneAtInit(GraphicalModel& gm, objcokus& rng, int choice, bool use_meta_feature = true);

    // save model meta-data, such as windowL, depthL, etc.
    virtual void saveMetaData(std::ostream& os) const;

    // load model meta-data, such as windowL, depthL, etc.
    virtual void loadMetaData(std::istream& is);

    // create a new sample from an instance.
    virtual ptr<GraphicalModel> makeSample(const Instance& instance, ptr<Corpus> corpus, objcokus* rng) const {
      throw "Model::makeSample not supported.";
    }

    // create a new sample with value equal to ground truth.
    virtual ptr<GraphicalModel> makeTruth(const Instance& instance, ptr<Corpus> corpus, objcokus* rng) const {
      throw "Model::makeTruth not supported.";
    }

    // create a new sample by copying an old one.
    virtual ptr<GraphicalModel> copySample(const GraphicalModel& gm) const {
      throw "Model::copySample not supported.";
    }

    // <deprecated> score a tag ?
    virtual double score(const GraphicalModel& gm);

    // evaluate the accuracy for POS tag aginst truth.
    // return 0: hit count.
    // return 1: pred count.
    std::tuple<int, int> evalPOS(const Tag& tag);
    // evaulate the F1 score for NER tag aginst truth.
    // return 0: hit count.
    // return 1: pred count.
    // return 2: truth count.
    std::tuple<int, int, int> evalNER(const Tag& tag);

    // return the Markov blanket of the node.
    // default: return the Markov blanket of node *id*
    virtual vec<int> markovBlanket(const GraphicalModel& gm, int pos) {
      vec<int> ret;
      for(int i = 0; i < gm.size(); i++) {
        if(i == pos) continue;
        ret.push_back(i);
      }
      return ret;
    }

    // return the nodes whose Markov blanket include the node.
    // default: return the Markov blanket of node *id*
    virtual vec<int> invMarkovBlanket(const GraphicalModel& gm, int pos) {
      return markovBlanket(gm, pos);
    }

    /* parameters */
    size_t T, B, Q;
    double testFrequency;
    double eta;
    std::vector<objcokus> rngs;
    ptr<Corpus> corpus;
    ParamPointer param, G2, stepsize;   // model.

    int time;

    /* IO */
    friend std::ostream& operator<<(std::ostream& os, const Model& model);
    friend std::istream& operator>>(std::istream& os, Model& model);

    ptr<XMLlog> xmllog;
    virtual void logArgs();

    /* const environment */

    enum Scoring {SCORING_NER, SCORING_ACCURACY, SCORING_LHOOD };
    Scoring scoring;

    void parseScoring(std::string scoring_str) {
      if(scoring_str == "Acc") scoring = SCORING_ACCURACY;
      else if(scoring_str == "NER") scoring = SCORING_NER;
      else if(scoring_str == "Lhood") scoring = SCORING_LHOOD;
      else throw "scoring method invalid";
    }

    std::string tostrScoring() const {
      switch(scoring) {
        case SCORING_ACCURACY:
          return "Acc";
        case SCORING_NER:
          return "NER";
        case SCORING_LHOOD:
          return "Lhood";
        default:
          throw "unknown scoring option";
      }
    }

    // options
    const boost::program_options::variables_map& vm;

  protected:
    void adagrad(ParamPointer gradient);
    void configStepsize(FeaturePointer gradient, double new_eta);

    int K;          // num of particle.
    int num_ob;     // current number of observations.
  };

  typedef std::shared_ptr<Model> ModelPtr;
  typedef std::function<FeaturePointer(ptr<Model> model, const GraphicalModel& gm, int pos)> FeatureExtractOne;
  typedef std::function<FeaturePointer(ptr<Model> model, const GraphicalModel& gm)> FeatureExtractAll;
  typedef std::function<vec<int>(ptr<Model> model, const GraphicalModel& gm, int pos)> MarkovBlanketGet;

  struct ModelSimple : public Model {
  public:
    ModelSimple(ptr<Corpus> corpus, const boost::program_options::variables_map& vm);

    ParamPointer gradient(const Instance& seq, TagVector* vec = nullptr, bool update_grad = true);
    ParamPointer gradient(const Instance& seq);

    virtual TagVector sample(const Instance& seq, bool argmax = false);
    virtual void sample(Tag& tag, int time, bool argmax = false);

    FeaturePointer extractFeatures(const Tag& tag, int pos);

    virtual void loadMetaData(std::istream& is);
    virtual void saveMetaData(std::ostream& os) const;

    virtual void logArgs();

    int windowL, depthL; // range of unigram features.
  };

  struct ModelCRFGibbs : public ModelSimple, public std::enable_shared_from_this<Model> {
  public:
    ModelCRFGibbs(ptr<Corpus> corpus, const boost::program_options::variables_map& vm);

    ParamPointer gradient(const Instance& seq, TagVector* vec = nullptr, bool update_grad = true);
    ParamPointer gradient(const Instance& seq);

    virtual TagVector sample(const Instance& seq, bool argmax = false);
    virtual void sample(Tag& tag, int time, bool argmax = false);

    double score(const GraphicalModel& tag);

    virtual void loadMetaData(std::istream& is);
    virtual void saveMetaData(std::ostream& os) const;

    virtual void logArgs();


    /* implement inferface for Gibbs sampling */
    virtual void sampleOne(GraphicalModel& tag, objcokus& rng, int choice, bool use_meta_feature = true);
    virtual void sampleOneAtInit(GraphicalModel& tag, objcokus& rng, int choice, bool use_meta_feature = true);

    /* implement interface for making samples */
    virtual ptr<GraphicalModel> makeSample(const Instance& instance, ptr<Corpus> corpus, objcokus* rng) const;
    virtual ptr<GraphicalModel> makeTruth(const Instance& instance, ptr<Corpus> corpus, objcokus* rng) const;
    virtual ptr<GraphicalModel> copySample(const GraphicalModel& gm) const;

    /* interface for feature extraction. */
    FeatureExtractOne extractFeatures;
    FeatureExtractOne extractFeaturesAtInit;
    FeatureExtractAll extractFeatAll;

    MarkovBlanketGet getMarkovBlanket;
    virtual vec<int> markovBlanket(const GraphicalModel& gm, int pos) {
      return getMarkovBlanket(shared_from_this(), gm, pos);
    }

    MarkovBlanketGet getInvMarkovBlanket;
    virtual vec<int> invMarkovBlanket(const GraphicalModel& gm, int pos) {
      return getInvMarkovBlanket(shared_from_this(), gm, pos);
    }

    /* properties */
    int factorL;

    /* annealing scheme. */
    string annealing;
    double temp, temp_decay, temp_magnify, temp_init;

  protected:
    // use Gibbs to sample <pos> with random number generator <rng> and feature extraction functional <feat_extract>
    // flags:
    //  grad_expect: add gradient based on expectation if set true.
    //  grad_sample: add gradient based on current sample if set true.
    //  meta_feature: do not change the meta features of tag if set true.
    ParamPointer proposeGibbs(Tag& tag, objcokus& rng, int pos, FeatureExtractOne feat_extract,
      bool grad_expect, bool grad_sample, bool meta_feature);

    void sampleOne(GraphicalModel& gm, objcokus& rng, int choice, FeatureExtractOne feat_extract, bool use_meta_feature = true);

    FeaturePointer extractFeaturesAll(const Tag& tag);

    void sampleOneSweep(Tag& tag, bool argmax = false);
  };

  struct MarkovTree;
  struct MarkovTreeNode;
}

#endif
